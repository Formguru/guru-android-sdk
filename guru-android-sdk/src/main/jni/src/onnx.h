//
// Created by Andrew Stahlman on 3/22/23.
//

#include <opencv2/core.hpp>

#ifndef GURUANDROIDSDK_ONNX_H
#define GURUANDROIDSDK_ONNX_H

class Point {
public:
    float x, y;

    Point(float _x, float _y) :
        x(_x), y(_y) {
    }
};

class KeyPoint {
public:
    Point p;
    float prob;

    KeyPoint(Point _p, float _prob) :
    p(_p), prob(_prob) {
    }
};

void init_model(const char*, size_t, int);
int infer_pose(cv::Mat&, cv::Rect&, std::vector<KeyPoint>&);


#endif //GURUANDROIDSDK_ONNX_H
