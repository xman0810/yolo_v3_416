#include "yolo_v3_detector.h"


static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s cvimodel image.jpg image_detected.jpg\n", argv[0]);
  printf("   or \n");
  printf("   %s prototxt caffemodel image.jpg image_detected.jpg\n", argv[0]);
}

int main(int argc, char **argv) {
  bool bTPU = true;
  char* prototxtFile = nullptr;
  char* caffemodelFile = nullptr;
  char* cvimodelFile = nullptr;
  char * inImageFile = nullptr;
  char * outImageFile = nullptr;

  if (argc == 4) {
    bTPU = true;
    cvimodelFile  = argv[1];
    inImageFile = argv[2];
    outImageFile = argv[3];
  } else if (argc == 5) {
    bTPU = false;
    prototxtFile = argv[1];
    caffemodelFile = argv[2];
    inImageFile = argv[3];
    outImageFile = argv[4];
  } else {
    usage(argv);
    exit(-1);
  }

  float mean[3] = {0.0f, 0.0f, 0.0f};
  float raw_scale = 1.0;
  float input_scale = 1.0f;

  // imread
  cv::Mat image;
  image = cv::imread(inImageFile);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
  }
  cv::Mat cloned = image.clone();
  detection det[MAX_DET];
  int32_t det_num = 0;
  YoloV3Detector * detector = nullptr;
  if (bTPU) {
    YoloV3Detector *tpuDetector = new YoloV3Detector(cvimodelFile,mean, raw_scale, input_scale);
    tpuDetector->doPreProcess(image);
    tpuDetector->doInference();
    tpuDetector->doPostProcess(cloned.rows, cloned.cols, det, MAX_DET, det_num);
    detector = tpuDetector;
  } else {
    YoloV3Detector *caffeDetector = new YoloV3Detector(caffemodelFile, prototxtFile, mean, raw_scale, input_scale);
    caffeDetector->caffeInference(image);
    caffeDetector->yolov3DetectionOutput();
    caffeDetector->doPostProcess(cloned.rows, cloned.cols, det, MAX_DET, det_num);
    detector = caffeDetector;
  }

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
    cv::putText(cloned, detector->getCocoLabel(det[i].cls), cv::Point(x1, y1),
                cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
  }
  delete detector;

  // save or show picture
  cv::imwrite(outImageFile, cloned);

  printf("------\n");
  printf("%d objects are detected\n", det_num);
  printf("------\n");

  return 0;
}
