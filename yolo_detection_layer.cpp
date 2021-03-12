#include "caffe/layers/yolo_detection_layer.hpp"

#include <sstream>

namespace caffe {

static inline float exp_fast(float x) {
  union {
    unsigned int i;
    float f;
  } v;
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);

  return v.f;
}

static inline float _sigmoid(float x, bool fast) {
  if (fast)
    return 1.0f / (1.0f + exp_fast(-x));
  else
    return 1.0f / (1.0f + exp(-x));
}

static inline float _softmax(float *probs, float *data, int input_stride,
    int num_of_class, int *max_cls, bool fast) {
  float x[num_of_class];
  float max_x = -INFINITY;
  float min_x = INFINITY;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = data[i * input_stride];
    if (x[i] > max_x) {
      max_x = x[i];
    }
    if (x[i] < min_x) {
      min_x = x[i];
    }
  }
  #define t (-100.0f)
  float exp_x[num_of_class];
  float sum = 0;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = x[i] - max_x;
    if (min_x < t)
      x[i] = x[i] / min_x * t;
    if (fast)
      exp_x[i] = exp_fast(x[i]);
    else
      exp_x[i] = exp(x[i]);
    sum += exp_x[i];
  }
  float max_prob = 0;
  for (int i = 0; i < num_of_class; i++) {
    probs[i] =exp_x[i] / sum;
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      *max_cls = i;
    }
  }
  return max_prob;
}

// feature in shape [3][5+80][grid_size][grid_size]
#define GET_INDEX(cell_idx, box_idx_in_cell, data_idx, num_cell, num_of_class) \
    (box_idx_in_cell * (num_of_class + 5) * num_cell + data_idx * num_cell + cell_idx)

static void process_feature(detection *det, int *det_idx, float *feature,
    std::vector<int> grid_size, float* anchor,
    std::vector<int> yolo_size, int num_of_class, float obj_threshold) {
  int yolo_w = yolo_size[1];
  int yolo_h = yolo_size[0];
  std::cout << "grid_size_h: " <<  grid_size[0] << std::endl;
  std::cout << "grid_size_w: " <<  grid_size[1] << std::endl;
  std::cout << "obj_threshold: " << obj_threshold << std::endl;
  int num_boxes_per_cell = 3;
  //assert(num_of_class == 80);

  // 255 = 3 * (5 + 80)
  // feature in shape [3][5+80][grid_size][grid_size]
  #define COORD_X_INDEX (0)
  #define COORD_Y_INDEX (1)
  #define COORD_W_INDEX (2)
  #define COORD_H_INDEX (3)
  #define CONF_INDEX    (4)
  #define CLS_INDEX     (5)
  int num_cell = grid_size[0] * grid_size[1];
  //int box_dim = 5 + num_of_class;

  int idx = *det_idx;
  int hit = 0, hit2 = 0;
  for (int i = 0; i < num_cell; i++) {
    for (int j = 0; j < num_boxes_per_cell; j++) {
      float box_confidence = _sigmoid(feature[GET_INDEX(i, j, CONF_INDEX, num_cell, num_of_class)], false);
      if (box_confidence < obj_threshold) {
        continue;
      }
      hit ++;
      float box_class_probs[80];
      int box_max_cls;
      float box_max_prob = _softmax(box_class_probs,
              &feature[GET_INDEX(i, j, CLS_INDEX, num_cell, num_of_class)],
              num_cell, num_of_class, &box_max_cls, false);
      float box_max_score = box_confidence * box_max_prob;
      if (box_max_score < obj_threshold) {
        continue;
      }
      // get coord now
      int grid_x = i % grid_size[1];
      int grid_y = i / grid_size[1];
      float box_x = _sigmoid(feature[GET_INDEX(i, j, COORD_X_INDEX, num_cell, num_of_class)], false);
      box_x += grid_x;
      box_x /= grid_size[1];
      float box_y = _sigmoid(feature[GET_INDEX(i, j, COORD_Y_INDEX, num_cell, num_of_class)], false);
      box_y += grid_y;
      box_y /= grid_size[0];
      // anchor is in shape [3][2]
      float box_w = exp(feature[GET_INDEX(i, j, COORD_W_INDEX, num_cell, num_of_class)]);
      box_w *= anchor[j*2];
      box_w /= yolo_w;
      float box_h = exp(feature[GET_INDEX(i, j, COORD_H_INDEX, num_cell, num_of_class)]);
      box_h *= anchor[j*2 + 1];
      box_h /= yolo_h;
      hit2 ++;
      //DBG("  hit2 %d, conf = %f, cls = %d, coord = [%f, %f, %f, %f]\n",
      //    hit2, box_max_score, box_max_cls, box_x, box_y, box_w, box_h);
      det[idx].bbox = box{box_x, box_y, box_w, box_h};
      det[idx].score = box_max_score;
      det[idx].cls = box_max_cls;
      idx++;
      assert(idx <= MAX_DET);
    }
  }
  *det_idx = idx;
}

// https://github.com/ChenYingpeng/caffe-yolov3/blob/master/box.cpp
static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1/2;
  float l2 = x2 - w2/2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1/2;
  float r2 = x2 + w2/2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if(w < 0 || h < 0) return 0;
  float area = w*h;
  return area;
}

static float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w*a.h + b.w*b.h - i;
  return u;
}

//
// more aboud iou
//   https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
// IoU = inter / (a + b - inter), can't handle enclosure issue
// GIoU, DIoU, CIoU?
//
static float box_iou(box a, box b) {
  return box_intersection(a, b)/box_union(a, b);
}

static void nms(detection *det, int num, float nms_threshold) {
  for(int i = 0; i < num; i++) {
    if (det[i].score == 0) {
      // erased already
      continue;
    }
    for(int j = i + 1; j < num; j++) {
      if (det[j].score == 0) {
        // erased already
        continue;
      }
      if (det[i].cls != det[j].cls) {
        // not the same class
        continue;
      }
      float iou = box_iou(det[i].bbox, det[j].bbox);
      assert(iou <= 1.0f);
      if (iou > nms_threshold) {
        // overlapped, select one to erase
        if (det[i].score < det[j].score) {
          det[i].score = 0;
        } else {
          det[j].score = 0;
        }
      }
    }
  }
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
  nms_threshold_ = this->layer_param_.yolo_detection_param().nms_threshold();
  obj_threshold_ = this->layer_param_.yolo_detection_param().obj_threshold();
  keep_topk_ = this->layer_param_.yolo_detection_param().keep_topk();
  net_input_h_ = this->layer_param_.yolo_detection_param().net_input_h();
  net_input_w_ = this->layer_param_.yolo_detection_param().net_input_w();
  tiny_ = this->layer_param_.yolo_detection_param().tiny();
  yolo_v4_ = this->layer_param_.yolo_detection_param().yolo_v4();
  class_num_ = this->layer_param_.yolo_detection_param().class_num();
  auto anchors = this->layer_param_.yolo_detection_param().anchors();
  anchors_.clear();
  std::istringstream iss(anchors);
  std::string s;
  while (std::getline(iss, s, ',')) {
    anchors_.push_back(atof(s.c_str()));
  }
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
{
    int batch = bottom[0]->num();
    top[0]->Reshape(batch, 1, keep_topk_, 6);
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
  auto top_data = top[0]->mutable_cpu_data();
  auto batch = top[0]->num();

  size_t bottom_count = bottom.size();

  float anchors[bottom_count][6];
  if (!tiny_) {
    if (anchors_.size() == 0) {
      if (yolo_v4_) {

        // order by prototext
        // refer yolov4.cfg for detail anchor setting
        anchors_ = {
          142, 110, 192, 243, 459, 401, // layer161-conv
          36, 75, 76, 55, 72, 146,// layer150-conv
          12, 16, 19, 36, 40, 28, // layer139-conv
        };
      }
      else {
        anchors_ = {
          10,13,   16,30,    33,23,      // layer106-conv (52*52)
          30,61,   62,45,    59,119,     // layer94-conv  (26*26)
          116,90,  156,198,  373,326     // layer82-conv  (13*13)
        };
      }
    }
  } else {
    if (anchors_.size() == 0) {
      anchors_ = {
          10,14,  23,27,    37,58,        // layer23-conv (26*26)
          81,82,  135,169,  344,319       // layer16-conv (13*13)
      };
    }
  }

  for (size_t i = 0; i < bottom_count; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      anchors[i][j] = anchors_[i*6+j];
    }
  }

  for (int b = 0; b < batch; ++b) {
    std::vector<std::vector<int>> grid_size;
    std::vector<std::vector<float>> features;

    for (int i = 0; i < bottom.size(); ++i) {
      std::vector<int> grid_hw{bottom[i]->shape()[2], bottom[i]->shape()[3]};
      grid_size.push_back(grid_hw);
      auto data = bottom[i]->cpu_data() + bottom[i]->offset(b);
      auto count = bottom[i]->count(1);
      std::vector<float> bottom_data(data, data + count);
      features.push_back(bottom_data);
    }

    detection det_raw[MAX_DET_RAW];
    detection dets[MAX_DET];
    int det_raw_idx = 0;
    for (int i = 0; i < features.size(); i++) {
      process_feature(det_raw, &det_raw_idx, features[i].data(), grid_size[i],
        &anchors[i][0], {net_input_h_, net_input_w_}, class_num_, obj_threshold_);

    }
    nms(det_raw, det_raw_idx, nms_threshold_);
    int det_idx = 0;
    for (int i = 0; i < det_raw_idx; i++) {
      if (det_raw[i].score > 0) {
        //DBG("keep cls %d, score %f, coord = [%f, %f, %f, %f], name %s\n",
        //    det_raw[i].cls, det_raw[i].score,
        //    det_raw[i].bbox.x, det_raw[i].bbox.y,
        //    det_raw[i].bbox.w, det_raw[i].bbox.h,
        //    coco_names[det_raw[i].cls]);
        dets[det_idx] = det_raw[i];
        det_idx ++;
      } else {
        //std::cout << "erased: " << det_raw[i].cls << std::endl;
      }
    }

    if (keep_topk_ > det_idx)
        keep_topk_ = det_idx;

    long long count = 0;
    auto batch_top_data = top_data + top[0]->offset(b);
    for(int i = 0; i < keep_topk_; ++i) {
      batch_top_data[count++] = dets[i].bbox.x;
      batch_top_data[count++] = dets[i].bbox.y;
      batch_top_data[count++] = dets[i].bbox.w;
      batch_top_data[count++] = dets[i].bbox.h;
      batch_top_data[count++] = dets[i].cls;
      batch_top_data[count++] = dets[i].score;
    }
  }
}

template <typename Dtype>
void YoloDetectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom)
{

}


#ifdef CPU_ONLY
STUB_GPU(YoloDetectionLayer);
#endif

INSTANTIATE_CLASS(YoloDetectionLayer);
REGISTER_LAYER_CLASS(YoloDetection);
}  // namespace caffe
