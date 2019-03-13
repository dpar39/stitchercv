

#include "Stitcher.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;

void Stitcher::processVideo(const std::string & inputVideoFile)
{
    cv::VideoCapture cap(inputVideoFile);

    auto x = cap.get(cv::CAP_PROP_FOURCC);
    if (!cap.isOpened())
    {
        return;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        splitInputImage(frame);
    }

    cap.release();
}

void Stitcher::splitInputImage(const cv::Mat & frame)
{
    const auto iWidth = frame.size[1] / 4;
    const auto height = frame.size[0];

    cv::Rect roi(0, 0, iWidth, height);
    for (int i = 0; i < 4; i++)
    {
        roi.x = i * iWidth;
        m_images[i] = frame(roi);

        detectAndDescribe(m_images[i]);
    }
}

void Stitcher::detectAndDescribe(const cv::Mat & framePart)
{
    cv::Mat grayPart;
    cv::Mat descriptors;
    cv::cvtColor(framePart, grayPart, cv::COLOR_BGR2GRAY);

    const auto sift = cv::xfeatures2d::SIFT::create();

    std::vector<KeyPoint> keyPoints;

    sift->detectAndCompute(framePart, cv::Mat(), keyPoints, descriptors);
}
