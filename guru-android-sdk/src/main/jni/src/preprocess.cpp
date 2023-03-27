//
// Created by Andrew Stahlman on 3/22/23.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "preprocess.h"
#include <android/bitmap.h>


cv::Mat normalize(cv::Mat image);

PreprocessedImage preprocess(cv::Mat image, int dest_width, int dest_height, cv::Rect* bounding_box) {
    /**
     * Scale the image to dest_width x dest_height with zero-padding as-needed.
     *
     * Note: the larger of image.width/dest_width and image.height/dest_height determines the scale.
     * The other dimension is the one that will be zero-padded.
     *
     * Source: https://stackoverflow.com/a/35598907/895769
     */

    bool has_good_bbox = (bounding_box != nullptr && (bounding_box->width >= 0.2 * image.cols) && (bounding_box->height >= 0.2 * image.rows));
    cv::Mat cropped;
    if (has_good_bbox) {
        cv::Mat(image, *bounding_box).copyTo(cropped);
    } else {
        cropped = image;
    }

    cv::Mat background = cv::Mat::zeros(dest_height, dest_width, CV_8UC4);
    float original_width = cropped.cols;
    float original_height = cropped.rows;

    cv::Mat transformation = cv::Mat::eye(3, 3, CV_32F);
    float scale_x = dest_width / original_width;
    float scale_y = dest_height / original_height;
    float x_translation = 0.0f;
    float y_translation = 0.0f;
    float scale;

    if (scale_x < scale_y) { // Scale on X, translate on Y
        scale = scale_x;
        y_translation = (dest_height - original_height * scale) / 2.0f;
    } else { // Scale on Y, translate on X
        scale = scale_y;
        x_translation = (dest_width - original_width * scale) / 2.0f;
    }

    transformation.at<float>(0, 0) = scale;
    transformation.at<float>(1, 1) = scale;
    transformation.at<float>(0, 2) = x_translation;
    transformation.at<float>(1, 2) = y_translation;

    cv::warpAffine(cropped, background, transformation.rowRange(0, 2), cv::Size(dest_width, dest_height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    cv::Mat feats = normalize(background);

    return PreprocessedImage(
            background,  // TODO: get rid of this
            feats,
            scale,
            x_translation,
            y_translation,
            has_good_bbox ? bounding_box->x : 0,
            has_good_bbox ? bounding_box->y : 0,
            image.cols,
            image.rows
    );
}

cv::Mat normalize(cv::Mat image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32FC3, 1.0/255.0);
    cv::Scalar meanMat = cv::Scalar(0.485f, 0.456f, 0.406f);
    cv::Scalar stdMat = cv::Scalar(0.229f, 0.224f, 0.225f);
    cv::subtract(normalized, meanMat, normalized);
    cv::divide(normalized, stdMat, normalized);
    return normalized;
}


