//
// Created by Andrew Stahlman on 3/22/23.
//

#include <opencv2/core.hpp>
#include <android/log.h>

#ifndef GURUANDROIDSDK_PREPROCESS_H
#define GURUANDROIDSDK_PREPROCESS_H


class PreprocessedImage {
public:
    cv::Mat bitmap;
    cv::Mat feats;
    float scale;
    float xPad;
    float yPad;
    int xOffset;
    int yOffset;
    int originalWidth;
    int originalHeight;

    PreprocessedImage(cv::Mat bitmap_, cv::Mat feats_, float scale_, float xPad_, float yPad_, int xOffset_, int yOffset_, int originalWidth_, int originalHeight_)
            : bitmap(bitmap_), feats(feats_), scale(scale_), xPad(xPad_), yPad(yPad_), xOffset(xOffset_), yOffset(yOffset_), originalWidth(originalWidth_), originalHeight(originalHeight_) {}
};

PreprocessedImage preprocess(cv::Mat image, int dest_width, int dest_height, cv::Rect* bounding_box);

#endif //GURUANDROIDSDK_PREPROCESS_H
