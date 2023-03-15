//
// Created by Andrew Stahlman on 3/8/23.
//

#include "pose.h"
#include <stdio.h>
#include <vector>
#include "net.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
static ncnn::Net posenet;

void init_model(const char* paramPath, const char* binPath) {
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;
    opt.use_vulkan_compute = true;
    posenet.opt = opt;
    if (posenet.load_param(paramPath)) {
        __android_log_print(ANDROID_LOG_ERROR, "guru_pose_inference", "load_model failed");
        exit(-1);
    }
    if (posenet.load_model(binPath)) {
        __android_log_print(ANDROID_LOG_ERROR, "guru_pose_inference", "load_model failed");
        exit(-1);
    }
}

int detect_posenet(ncnn::Mat& rgb, std::vector<KeyPoint>& keypoints)
{
    // transforms.ToTensor(),
    // transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    // R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
    // G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
    // B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    rgb.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = posenet.create_extractor();
    ex.input("input", rgb);

    ncnn::Mat out;
    ex.extract("output", out);

    // resolve point from heatmap
    keypoints.clear();
    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for (int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        keypoint.p = { max_x / (float)out.w, max_y / (float)out.h };
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }

    return 0;
}