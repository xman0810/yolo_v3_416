#include "yolo_v3_detector.h"
#define MAX_DET 200
#define MAX_DET_RAW 500

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

static const char *coco_names[] = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"};

void YoloV3Detector::init(float *mean, float raw_scale, float input_scale) {
  this->mean[0] = mean[0];
  this->mean[1] = mean[1];
  this->mean[2] = mean[2];

  this->raw_scale = raw_scale;
  this->input_scale = input_scale;
  batch = 1;
  keep_topk = 200;
  nms_threshold = 0.4;
  obj_threshold = 0.5;
  class_num = 80;
}

YoloV3Detector::YoloV3Detector(const char * caffemodel_file, const char *prototxt_file,
               float *mean, float raw_scale, float input_scale) {
  init(mean, raw_scale, input_scale);
  net.reset(new caffe::Net<float>(prototxt_file, caffe::TEST));
  net->CopyTrainedLayersFrom(caffemodel_file);
  caffe::Blob<float> *input_data_blobs = net->input_blobs()[0];
  for (int i = 0; i < input_data_blobs->num_axes(); i++) {
    shape[i] = input_data_blobs->shape(i);
  }

  height = shape[2];
  width = shape[3];
  output_num = net->num_outputs();
  layer106_data = nullptr;
  layer94_data = nullptr;
  layer82_data  = nullptr;
  output_data = nullptr;
  output_blob = new caffe::Blob<float>(batch, 1, keep_topk, 6);
}

YoloV3Detector::YoloV3Detector(const char *model_file, float *mean, float raw_scale, float input_scale, bool nhwc) {
  init(mean, raw_scale, input_scale);
  int ret = CVI_NN_RegisterModel(model_file, &model);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    exit(1);
  }
  printf("CVI_NN_RegisterModel succeeded\n");

  // get input output tensors
  CVI_NN_SetConfig(model, OPTION_BATCH_SIZE, 1);
  CVI_NN_SetConfig(model, OPTION_SKIP_PREPROCESS, true);
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors,
                               &output_num);

  input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
  assert(input);
  output = CVI_NN_GetTensorByName("output", output_tensors, output_num);
  assert(output);

  qscale = CVI_NN_TensorQuantScale(input);
  auto cvi_shape = CVI_NN_TensorShape(input);
  if (nhwc) {
    height = cvi_shape.dim[1];
    width = cvi_shape.dim[2];
  } else {
    height = cvi_shape.dim[2];
    width = cvi_shape.dim[3];
  }
  printf("shape info: dim_size=%d,", cvi_shape.dim_size);
  for(int i=0;i<cvi_shape.dim_size;i++){
    shape[i] = cvi_shape.dim[i];
    printf("dim[%d]=%d,", i, shape[i]);
  }
  batch = 1;
  keep_topk = 200;
}

YoloV3Detector::~YoloV3Detector() {
  if (model) {
    CVI_NN_CleanupModel(model);
    printf("CVI_NN_CleanupModel succeeded\n");
  }
}

void YoloV3Detector::doCaffePreProcess(cv::Mat &image, cv::Mat *channels) {
  // resize & letterbox
  int ih = image.rows;
  int iw = image.cols;
  int oh = shape[2];
  int ow = shape[3];
  double resize_scale = std::min((double)oh / ih, (double)ow / iw);
  int nh = (int)(ih * resize_scale);
  int nw = (int)(iw * resize_scale);
  cv::resize(image, image, cv::Size(nw, nh));
  int top = (oh - nh) / 2;
  int bottom = (oh - nh) - top;
  int left = (ow - nw) / 2;
  int right = (ow - nw) - left;
  cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0));

  for (int i = 0; i < 3; i++) {
      channels[i] = cv::Mat(oh, ow, CV_32FC1);
  }
  cv::split(image, channels);
  // normalize
  for (int i = 0; i < 3; i++) {
    channels[i].convertTo(channels[i], CV_32FC1, raw_scale / 255.0, -mean[i]);
    channels[i].convertTo(channels[i], CV_32FC1, input_scale, 0);
  }
}

void YoloV3Detector::caffeInference(cv::Mat &image) {
  cv::Mat channels[3];
  doCaffePreProcess(image, channels);
  caffe::Blob<float> *input_data_blobs = net->input_blobs()[0];
  float *f32inputdata = (float*)input_data_blobs->mutable_cpu_data();
  // BRG -> RGB
  for (int i = 0; i < 3; i++) {
    int chlIdx = i;
    if (i == 0)
      chlIdx = 2;
    if (i == 2)
      chlIdx = 0;

    memcpy(f32inputdata, channels[chlIdx].data, input_data_blobs->width()*input_data_blobs->height() * sizeof(float));
    f32inputdata += input_data_blobs->width()*input_data_blobs->height();
  }

  net->Forward();
  if (net->has_blob("layer106-conv")) {
    layer106_blob = net->blob_by_name("layer106-conv").get();
    layer106_data = (float *)(layer106_blob->cpu_data());
  }
  if (net->has_blob("layer94-conv")) {
    layer94_blob = net->blob_by_name("layer94-conv").get();
    layer94_data = (float *)(layer94_blob->cpu_data());
  }
  if (net->has_blob("layer82-conv")) {
    layer82_blob = net->blob_by_name("layer82-conv").get();
    layer82_data = (float *)(layer82_blob->cpu_data());
  }
}

void YoloV3Detector::doPreProcess(cv::Mat &image) {
  // resize & letterbox
  int ih = image.rows;
  int iw = image.cols;
  int oh = height;
  int ow = width;
  double resize_scale = std::min((double)oh / ih, (double)ow / iw);
  int nh = (int)(ih * resize_scale);
  int nw = (int)(iw * resize_scale);
  cv::resize(image, image, cv::Size(nw, nh));
  int top = (oh - nh) / 2;
  int bottom = (oh - nh) - top;
  int left = (ow - nw) / 2;
  int right = (ow - nw) - left;
  cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0));
  // split
  cv::Mat channels[3];
  for (int i = 0; i < 3; i++) {
    channels[i] = cv::Mat(height, width, CV_8SC1);
  }
  cv::split(image, channels);

  // normalize
  for (int i = 0; i < 3; i++) {
    channels[i].convertTo(channels[i], CV_32FC1, raw_scale / 255.0, -mean[i]);
    channels[i].convertTo(channels[i], CV_32FC1, input_scale, 0);
  }

  for (int i = 0; i < 3; i++) {
    channels[i].convertTo(channels[i], CV_8SC1, qscale, 0);
  }

  // BGR -> RGB & fill data
  int8_t *ptr = (int8_t *)CVI_NN_TensorPtr(input);
  int channel_size = height * width;
  memcpy(ptr + 2 * channel_size, channels[0].data, channel_size);
  memcpy(ptr + channel_size, channels[1].data, channel_size);
  memcpy(ptr, channels[2].data, channel_size);
}

void YoloV3Detector::doPreProccess_ResizeOnly(cv::Mat &image) {
  // resize & letterbox
  int ih = image.rows;
  int iw = image.cols;
  int oh = height;
  int ow = width;
  double scale = std::min((double)oh / ih, (double)ow / iw);
  int nh = (int)(ih * scale);
  int nw = (int)(iw * scale);
  cv::resize(image, image, cv::Size(nw, nh));
  int top = (oh - nh) / 2;
  int bottom = (oh - nh) - top;
  int left = (ow - nw) / 2;
  int right = (ow - nw) - left;
  cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0));
  memcpy(CVI_NN_TensorPtr(input), image.data, CVI_NN_TensorSize(input));
}

void YoloV3Detector::yolov3DetectionOutput() {
  std::vector<caffe::Blob<float>*> bottom;
  std::vector<caffe::Blob<float>*> top;
  bottom.push_back(layer106_blob);
  bottom.push_back(layer94_blob);
  bottom.push_back(layer82_blob);
  top.push_back(output_blob);

  auto top_data = top[0]->mutable_cpu_data();
  auto batch = top[0]->num();

  size_t bottom_count = bottom.size();

  float anchors[bottom_count][6];
  std::vector<float> yolov3_anchors = {
    10,13,   16,30,    33,23,      // layer106-conv (52*52)
    30,61,   62,45,    59,119,     // layer94-conv  (26*26)
    116,90,  156,198,  373,326     // layer82-conv  (13*13)
  };

  for (size_t i = 0; i < bottom_count; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      anchors[i][j] = yolov3_anchors[i*6+j];
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
        &anchors[i][0], {height, width}, class_num, obj_threshold);

    }
    nms(det_raw, det_raw_idx, nms_threshold);
    int det_idx = 0;
    for (int i = 0; i < det_raw_idx; i++) {
      if (det_raw[i].score > 0) {
        printf("keep cls %d, score %f, coord = [%f, %f, %f, %f], name %s\n",
            det_raw[i].cls, det_raw[i].score,
            det_raw[i].bbox.x, det_raw[i].bbox.y,
            det_raw[i].bbox.w, det_raw[i].bbox.h,
            coco_names[det_raw[i].cls]);
        dets[det_idx] = det_raw[i];
        det_idx ++;
      } else {
        //std::cout << "erased: " << det_raw[i].cls << std::endl;
      }
    }

    if (keep_topk > det_idx)
        keep_topk = det_idx;

    long long count = 0;
    auto batch_top_data = top_data + top[0]->offset(b);
    for(int i = 0; i < keep_topk; ++i) {
      batch_top_data[count++] = dets[i].bbox.x;
      batch_top_data[count++] = dets[i].bbox.y;
      batch_top_data[count++] = dets[i].bbox.w;
      batch_top_data[count++] = dets[i].bbox.h;
      batch_top_data[count++] = dets[i].cls;
      batch_top_data[count++] = dets[i].score;
    }
  }
  output_data = (float *)(output_blob->cpu_data());
}


void YoloV3Detector::doInference() {
  // run inference
  CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  output_data = (float *)CVI_NN_TensorPtr(output);
}

void YoloV3Detector::doPostProcess(int32_t image_h, int32_t image_w, detection dets[],
                                   int32_t max_det_num, int32_t &det_num) {
  det_num = 0;
  for (int i = 0; i < max_det_num; ++i) {
    // filter real det with score > 0
    if (output_data[i * 6 + 5] > 0) {
      // output: [x,y,w,h,cls,score]
      dets[det_num].bbox.x = output_data[i * 6 + 0];
      dets[det_num].bbox.y = output_data[i * 6 + 1];
      dets[det_num].bbox.w = output_data[i * 6 + 2];
      dets[det_num].bbox.h = output_data[i * 6 + 3];
      dets[det_num].cls = output_data[i * 6 + 4];
      dets[det_num].score = output_data[i * 6 + 5];
      det_num++;
    }
  }
  printf("get detection num: %d\n", det_num);
  // correct box with origin image size
  correct_yolo_boxes(dets, det_num, image_h, image_w, false);
}

void YoloV3Detector::correct_yolo_boxes(detection *dets, int det_num, int image_h,
                                        int image_w, bool relative_position) {
  int i;
  int restored_w = 0;
  int restored_h = 0;
  if (((float)width / image_w) < ((float)height / image_h)) {
    restored_w = width;
    restored_h = (image_h * width) / image_w;
  } else {
    restored_h = height;
    restored_w = (image_w * height) / image_h;
  }
  for (i = 0; i < det_num; ++i) {
    box b = dets[i].bbox;
    b.x = (b.x - (width - restored_w) / 2. / width) /
          ((float)restored_w / width);
    b.y = (b.y - (height - restored_h) / 2. / height) /
          ((float)restored_h / height);
    b.w *= (float)width / restored_w;
    b.h *= (float)height / restored_h;
    if (!relative_position) {
      b.x *= image_w;
      b.w *= image_w;
      b.y *= image_h;
      b.h *= image_h;
    }
    dets[i].bbox = b;
  }
}

const char *YoloV3Detector::getCocoLabel(int cls) {
  return coco_names[cls];
}
