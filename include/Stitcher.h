#pragma once

#include <opencv2/core/core.hpp>

namespace cv::xfeatures2d
{
class SIFT;
}

class Stitcher
{

public:
    Stitcher();

    void processVideo(const std::string & inputVideoFile);

private:
    void splitInputImage(const cv::Mat & frame);

    cv::Mat computeHomography(const cv::Mat & leftImage, const cv::Mat & rightImage);

    cv::Mat m_images[4];

    cv::Mat m_leftImageMask;

    cv::Mat m_rightImageMask;

    cv::Ptr<cv::xfeatures2d::SIFT> m_sift;
};
