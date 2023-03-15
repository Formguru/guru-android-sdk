//
// Created by Andrew Stahlman on 3/8/23.
//

#ifndef GURU_POSE_INFERENCE_POSE_H
#define GURU_POSE_INFERENCE_POSE_H

#include <net.h>

struct Point {
    float x, y;
};

struct KeyPoint
{
    Point p;
    float prob;
};

void init_model(const char* paramPath, const char* binPath);
int detect_posenet(ncnn::Mat& rgb, std::vector<KeyPoint>& keypoints);

#endif //GURU_POSE_INFERENCE_POSE_H
