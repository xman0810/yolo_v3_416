#include "yolo_v3_detector.h"

// #define SAVE_FILE_FOR_DEBUG
// #define DO_IMSHOW

static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s cvimodel image.jpg image_detected.jpg\n", argv[0]);
}

int main(int argc, char **argv) {
  int ret = 0;
  CVI_MODEL_HANDLE model;

  if (argc != 4) {
    usage(argv);
    exit(-1);
  }

  float mean[3] = {104.0f, 117.0f, 123.0f};
  float raw_scale = 255;
  float input_scale = 1.0f;
  YoloV3Detector detector(argv[1], mean, raw_scale, input_scale, true);

  // imread
  cv::Mat image;
  image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
  }
  cv::Mat cloned = image.clone();

  detection det[MAX_DET];
  int32_t det_num = 0;

  detector.doPreProccess_ResizeOnly(image);
  detector.doInference();
  detector.doPostProcess(cloned.rows, cloned.cols, det, MAX_DET, det_num);

  // draw bbox on image
  for (int i = 0; i < det_num; i++) {
    box b = det[i].bbox;
    // xywh2xyxy
    int x1 = (b.x - b.w / 2);
    int y1 = (b.y - b.h / 2);
    int x2 = (b.x + b.w / 2);
    int y2 = (b.y + b.h / 2);
    cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0),
                  3, 8, 0);
    cv::putText(cloned, detector.getCocoLabel(det[i].cls), cv::Point(x1, y1),
                cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  }

  // save or show picture
  cv::imwrite(argv[3], cloned);

  printf("------\n");
  printf("%d objects are detected\n", det_num);
  printf("------\n");

  return 0;
}
