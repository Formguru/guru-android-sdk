//
// Created by Andrew Stahlman on 3/9/23.
//

#include <jni.h>
#include "pose.h"
#include "benchmark.h"
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <net.h>
#include <vector>


extern "C"
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "guru_pose_inference", "JNI_OnLoad");
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_6;
}

extern "C"
JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "guru_pose_inference", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

extern "C"
JNIEXPORT jfloatArray JNICALL Java_ai_getguru_androidsdk_NcnnPoseEstimator_detectPose(JNIEnv *env, jobject thiz, jobject bitmap) {

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        exit(-1);
    }
    int target_width = 192;
    int target_height = 256;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, target_width, target_height);
    std::vector<KeyPoint> keypoints;
    int result = detect_posenet(in, keypoints);
    if (result != 0) {
        exit(-1);
    }

    double elapsed = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "guru_pose_inference", "Inference took %.2fms", elapsed);

    // intentionally drop the feet keypoints so that the viz is less cluttered...

    jfloat output[keypoints.size() * 3];
    uint_t k = 0;
    for (KeyPoint kpt: keypoints) {
        output[3*k] = kpt.p.x;
        output[3*k + 1] = kpt.p.y;
        output[3*k + 2] = kpt.prob;
        k++;
    }
    jfloatArray arr = (*env).functions->NewFloatArray(env, keypoints.size() * 3);
    env->SetFloatArrayRegion(arr, 0, keypoints.size() * 3, output);
    return arr;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_getguru_androidsdk_NcnnPoseEstimator_initModel(JNIEnv *env, jobject thiz, jstring paramPath, jstring binPath) {
    init_model(
            env->GetStringUTFChars(paramPath, 0),
            env->GetStringUTFChars(binPath, 0)
    );
}