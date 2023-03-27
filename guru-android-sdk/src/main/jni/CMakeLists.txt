project(guru_pose_inference)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni")
find_package(OpenCV REQUIRED)

add_library(libonnxruntime STATIC IMPORTED)
set_target_properties(
        libonnxruntime
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/onnxruntime/${ANDROID_ABI}/libonnxruntime.so
)
include_directories(${CMAKE_SOURCE_DIR}/onnxruntime/headers)

add_library(
        guru_pose_inference
        SHARED
        src/pose_jni.cpp src/onnx.cpp src/preprocess.cpp
)
target_link_libraries(
        guru_pose_inference
        libonnxruntime
        -ljnigraphics
        ${OpenCV_LIBS}
)