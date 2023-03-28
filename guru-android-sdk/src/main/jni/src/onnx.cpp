//
// Created by Andrew Stahlman on 3/22/23.
//

#include "onnx.h"
#include "preprocess.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn/dnn.hpp>

const int INPUT_WIDTH = 192;
const int INPUT_HEIGHT = 256;
const int HEATMAP_WIDTH = 48;
const int HEATMAP_HEIGHT = 64;

Ort::Env env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
std::unique_ptr<Ort::Session> session_;
const std::vector<const char*> input_names { "input" };
const std::vector<const char*> output_names { "output" };
const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
const std::vector<int64_t> tensor_dims = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

Ort::AllocatorWithDefaultOptions allocator;

void init_model(const char* model_bytes, size_t model_data_len, int num_processors) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_processors);
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    session_ = std::make_unique<Ort::Session>(env_, model_bytes, model_data_len, session_options);
}

std::tuple<float, float, float> argmax(const Ort::Value&, int);
void post_process(const Ort::Value&, const PreprocessedImage&, std::vector<KeyPoint>&);

// Useful for visualizing intermediate outputs, i.e., write to /sdcard and then `adb pull`
void save_mat_to_file(const cv::Mat& mat, const std::string& file_path) {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    bool b = cv::imwrite(file_path, mat, params);
    if (!b) {
        __android_log_print(ANDROID_LOG_ERROR, "guru_pose_inference", "Write failed!");
    }
}

int infer_pose(cv::Mat& img, cv::Rect& bbox, std::vector<KeyPoint>& keypoints) {
    const PreprocessedImage &image = preprocess(img, INPUT_WIDTH, INPUT_HEIGHT, &bbox);
    cv::Mat nchw = cv::dnn::blobFromImage(image.feats);

    const size_t input_tensor_size = image.feats.total() * image.feats.channels();
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            (float*)nchw.ptr<float>(0),
            input_tensor_size,
            tensor_dims.data(),
            tensor_dims.size()
    ));

    auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            1,
            output_names.data(),
            1);

    const Ort::Value& heatmaps = output_tensors.at(0);
    post_process(heatmaps, image, keypoints);
    return 0;
}

std::tuple<float, float, float> argmax(const Ort::Value& heatmaps, int k) {
    const float* vals = heatmaps.GetTensorData<float>();
    float max_score = std::numeric_limits<float>::lowest();
    int best_row, best_col = 0;
    int k_start = k * HEATMAP_HEIGHT * HEATMAP_WIDTH;
    for (int64_t row = 0; row < HEATMAP_HEIGHT; row++) {
        for (int64_t col = 0; col < HEATMAP_WIDTH; col++) {
            float score = vals[k_start + (row * HEATMAP_WIDTH) + col];
            if (score > max_score) {
                max_score = score;
                best_row = row;
                best_col = col;
            }
        }
    }

    return std::tuple<float, float, float> { best_col, best_row, max_score };
}

void post_process(const Ort::Value& heatmaps, const PreprocessedImage& preprocessed, std::vector<KeyPoint>& keypoints) {
    auto shape = heatmaps.GetTensorTypeAndShapeInfo().GetShape();
    auto K = shape.at(1);

    std::vector<int64_t> expected_shape = {1, K, HEATMAP_HEIGHT, HEATMAP_WIDTH };
    assert(expected_shape == shape);

    for (int k = 0; k < K; k++) {
        auto xy_score = argmax(heatmaps, k);
        float x = std::get<0>(xy_score);
        float y = std::get<1>(xy_score);
        float score = std::get<2>(xy_score);

        x *= (INPUT_WIDTH / (float)HEATMAP_WIDTH);
        x -= preprocessed.xPad;
        x /= preprocessed.scale;
        x += preprocessed.xOffset;
        x /= preprocessed.originalWidth;

        y *= (INPUT_HEIGHT / (float)HEATMAP_HEIGHT);
        y -= preprocessed.yPad;
        y /= preprocessed.scale;
        y += preprocessed.yOffset;
        y /= preprocessed.originalHeight;

        keypoints.push_back(KeyPoint(Point(x, y), score));
    }
}
