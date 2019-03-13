#pragma once

#include <opencv2/core/core.hpp>

class Stitcher
{

public:
    void processVideo(const std::string & inputVideoFile);

private:
    void splitInputImage(const cv::Mat & frame);

    cv::Mat m_images[4];
};
