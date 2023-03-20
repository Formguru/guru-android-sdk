project(guru_pose_inference)

cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(guru_pose_inference SHARED src/pose_jni.cpp src/pose.cpp)
target_link_libraries(
        guru_pose_inference
        ncnn
)