#pragma once

#include <opencv2/core/core.hpp>

class Stitcher
{

public:
    void processVideo(const std::string & inputVideoFile);

private:
    void splitInputImage(const cv::Mat & frame);

    void detectAndDescribe(const cv::Mat & framePart);

    cv::Mat m_images[4];
};
