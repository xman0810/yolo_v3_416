cmake_minimum_required(VERSION 2.8.0)

project(cvi_sample_detector C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_BUILD_TYPE Debug)

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH to point to the TPU_SDK installation")
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

set(CMAKE_CXX_FLAGS "-DCPU_ONLY=1")
include_directories(${CAFFE_PATH}/include)
link_directories(${CAFFE_PATH}/lib)

if(NOT DEFINED OPENCV_PATH)
  message(FATAL_ERROR "Please set OPENCV_PATH to point to the opencvn installation")
endif()
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
endif()

set(OPENCV_LIBS ${OPENCV_LIBS} opencv_core opencv_imgcodecs opencv_imgproc)
if(NOT CMAKE_CROSSCOMPILING)
  set(OPENCV_LIBS ${OPENCV_LIBS} opencv_highgui)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread glog z)

add_executable(cvi_sample_detector_yolo_v3
    detector.cpp
    yolo_v3_detector.cpp)
target_link_libraries(cvi_sample_detector_yolo_v3
    ${CVI_LIBS}
    ${OPENCV_LIBS}
    ${EXTRA_LIBS}
    ${CAFFE_PATH}/lib/libcaffe-d.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so)
install(TARGETS cvi_sample_detector_yolo_v3
    cvi_sample_detector_yolo_v3 DESTINATION bin)

add_executable(cvi_sample_detector_yolo_v3_fused_preprocess
    detector_with_fused_preprocess.cpp
    yolo_v3_detector.cpp)
target_link_libraries(cvi_sample_detector_yolo_v3_fused_preprocess
    ${CVI_LIBS}
    ${OPENCV_LIBS}
    ${EXTRA_LIBS}
    ${CAFFE_PATH}/lib/libcaffe-d.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so)
install(TARGETS cvi_sample_detector_yolo_v3_fused_preprocess
    cvi_sample_detector_yolo_v3_fused_preprocess DESTINATION bin)

install(FILES dog.jpg DESTINATION data)
