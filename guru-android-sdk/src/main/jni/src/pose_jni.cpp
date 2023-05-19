//
// Created by Andrew Stahlman on 3/9/23.
//

#include <jni.h>
#include "preprocess.h"
#include "onnx.h"
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <vector>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

static jfieldID bbox_x1_field;
static jfieldID bbox_y1_field;
static jfieldID bbox_x2_field;
static jfieldID bbox_y2_field;

static jint JNI_VERSION = JNI_VERSION_1_6;


cv::Mat bitmap_to_mat(JNIEnv *env, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    cv::Mat mat;

    // Get bitmap information
    int ret = AndroidBitmap_getInfo(env, bitmap, &info);
    if (ret != ANDROID_BITMAP_RESULT_SUCCESS) {
        __android_log_print(ANDROID_LOG_FATAL, "guru_pose_inference", "Failed to get bitmap info");
        exit(-1);
    }

    // Check bitmap format
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_FATAL, "guru_pose_inference", "Bitmap format is not RGBA_8888");
        exit(-1);
    }

    // Lock the bitmap pixels to access them
    ret = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (ret != ANDROID_BITMAP_RESULT_SUCCESS) {
        __android_log_print(ANDROID_LOG_FATAL, "guru_pose_inference", "Failed to lock pixels");
        exit(-1);
    }

    // Create a Mat object with the bitmap data
    mat = cv::Mat(info.height, info.width, CV_8UC4, pixels);

    // Convert RGBA to RGB
    cv::cvtColor(mat, mat, cv::COLOR_RGBA2RGB);

    // Unlock the bitmap pixels
    AndroidBitmap_unlockPixels(env, bitmap);

    return mat;
}

extern "C"
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "guru_pose_inference", "JNI_OnLoad");

    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    jclass clazz = env->FindClass("ai/getguru/androidsdk/BoundingBox");
    bbox_x1_field = env->GetFieldID(clazz, "x1", "F");
    bbox_y1_field = env->GetFieldID(clazz, "y1", "F");
    bbox_x2_field = env->GetFieldID(clazz, "x2", "F");
    bbox_y2_field = env->GetFieldID(clazz, "y2", "F");
    env->DeleteLocalRef(clazz);

    return JNI_VERSION_1_6;
}

extern "C"
JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "guru_pose_inference", "JNI_OnUnload");
}

cv::Rect to_bbox(JNIEnv *env, const cv::Mat& img, jobject bbox) {
    jfloat x1 = env->GetFloatField(bbox, bbox_x1_field) * img.cols;
    jfloat y1 = env->GetFloatField(bbox, bbox_y1_field) * img.rows;
    jfloat x2 = env->GetFloatField(bbox, bbox_x2_field) * img.cols;
    jfloat y2 = env->GetFloatField(bbox, bbox_y2_field) * img.rows;
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

extern "C"
JNIEXPORT void Java_ai_getguru_androidsdk_OnnxPoseEstimator_initFromFile(JNIEnv *env, jobject thiz, jstring model_path, jint numProcessors) {
    const char *path = env->GetStringUTFChars(model_path, 0);
    std::ifstream infile(std::string(path), std::ios_base::binary);
    std::string buffer((std::istreambuf_iterator<char>(infile)),
                         std::istreambuf_iterator<char>());
    init_model(buffer.c_str(), buffer.size(), numProcessors);
}

extern "C"
JNIEXPORT void Java_ai_getguru_androidsdk_OnnxPoseEstimator_initFromAsset(JNIEnv *env, jobject thiz, jstring model_path, jobject asset_mgr, jint numProcessors) {
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_mgr);
    const char *path = env->GetStringUTFChars(model_path, 0);
    AAsset* file = AAssetManager_open(mgr, path, AASSET_MODE_BUFFER);
    size_t file_len = AAsset_getLength(file);
    char* model_bytes = new char[file_len+1];
    AAsset_read(file, model_bytes, file_len);
    model_bytes[file_len] = '\0';
    init_model(model_bytes, file_len, numProcessors);
    delete [] model_bytes;
}

jfloatArray do_inference(JNIEnv *env, cv::Mat& img, cv::Rect& bbox) {
    std::vector<KeyPoint> keypoints;
    infer_pose(img, bbox, keypoints);
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
JNIEXPORT jfloatArray Java_ai_getguru_androidsdk_OnnxPoseEstimator_detectPose(JNIEnv *env, jobject thiz, jobject bitmap, jobject bbox) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_FATAL, "guru_pose_inference", "Unsupported format for Bitmap: %d", info.format);
        exit(-1);
    }
    cv::Mat img = bitmap_to_mat(env, bitmap);
    cv::Rect rect = to_bbox(env, img, bbox);
    return do_inference(env, img, rect);
}

extern "C"
JNIEXPORT jfloatArray Java_ai_getguru_androidsdk_OnnxPoseEstimator_detectPoseFromNv21(JNIEnv *env, jobject thiz, jbyteArray nv21, jint width, jint height, jint rotation, jobject bbox) {
    int len = env->GetArrayLength(nv21);
    unsigned char *buf = new unsigned char[len];
    env->GetByteArrayRegion(nv21, 0, len, reinterpret_cast<jbyte *>(buf));

    cv::Mat img(height + height / 2, width, CV_8UC1, (uchar *) buf);
    cv::cvtColor(img, img, cv::COLOR_YUV2RGBA_NV21);

    cv::Mat final_img;
    if (rotation == 0) {
        final_img = img;
    } else {
        int rotate_flag;
        if (rotation == 90) {
            rotate_flag = 0;
        } else if (rotation == 180) {
            rotate_flag = 1;
        } else if (rotation == 270) {
            rotate_flag = 2;
        } else {
            __android_log_print(ANDROID_LOG_FATAL, "guru_pose_inference", "Unsupported rotation: %d", rotation);
            exit(1);
        }
        cv::rotate(img, final_img, rotate_flag);
    }

    cv::Rect rect = to_bbox(env, final_img, bbox);
    auto results = do_inference(env, final_img, rect);
    delete[] buf;
    return results;
}