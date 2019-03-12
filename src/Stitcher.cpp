

#include "Stitcher.h"

#include <opencv2/videoio/videoio.hpp>

void Stitcher::processVideo(const std::string& inputVideoFile)
{
    cv::VideoCapture cap(inputVideoFile);

    auto x = cap.get(cv::CAP_PROP_FOURCC);
    if (!cap.isOpened()) {
        return;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        splitInputImage(frame);
    }

    cap.release();
}

void Stitcher::splitInputImage(const cv::Mat& frame)
{
    for (int i = 0; i < 4; i++) {
    }
}
