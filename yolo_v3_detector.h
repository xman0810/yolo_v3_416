#ifndef YOLO_V3_DETECTOR_H
#define YOLO_V3_DETECTOR_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <caffe/caffe.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"

#define MAX_DET 200

typedef struct {
  float x, y, w, h;
} box;

typedef struct {
  box bbox;
  int cls;
  float score;
} detection;

class YoloV3Detector {
public:

  void init(float *mean, float raw_scale, float input_scale);
  YoloV3Detector(const char *model_file, float *mean, float raw_scale, float input_scale, bool nhwc = false);
  YoloV3Detector(const char * caffemodel_file, const char *prototxt_file,
               float *mean, float raw_scale, float input_scale);
  ~YoloV3Detector();

  void doCaffePreProcess(cv::Mat &image, cv::Mat *channels);
  void doPreProcess(cv::Mat &image);
  void doPreProccess_ResizeOnly(cv::Mat &image);
  void doInference();
  void caffeInference(cv::Mat &image);
  void doPostProcess(int32_t image_h, int32_t image_w, detection det[],
                     int32_t max_det_num, int32_t &det_num);

  void yolov3DetectionOutput();
  const char *getCocoLabel(int32_t cls);

private:
  int batch;
  int keep_topk;
  float nms_threshold;
  float obj_threshold;
  int class_num;
  void correct_yolo_boxes(detection *dets, int det_num, int image_h, int image_w,
                          bool relative_position);

public:
  CVI_TENSOR *input;
  CVI_TENSOR *output;
  std::shared_ptr<caffe::Net<float> > net;
  caffe::Blob<float>* layer106_blob;
  caffe::Blob<float>* layer94_blob;
  caffe::Blob<float>* layer82_blob;
  caffe::Blob<float>* output_blob;
  float *layer106_data;
  float *layer94_data;
  float *layer82_data;
  float *output_data;

private:
  CVI_MODEL_HANDLE model = nullptr;
  CVI_TENSOR *input_tensors;
  CVI_TENSOR *output_tensors;
  int32_t input_num;
  int32_t output_num;
  int32_t shape[6];
  int32_t height;
  int32_t width;
  float qscale;
  float mean[3];
  float input_scale;
  float raw_scale;
};

#endif
