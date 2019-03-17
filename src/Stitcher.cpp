

#include "Stitcher.h"

#include <cmath>
#include <filesystem>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;

Stitcher::Stitcher()
{
    m_sift = cv::xfeatures2d::SIFT::create();
}

void Stitcher::processVideo(const std::string & inputVideoFile)
{
    cv::VideoCapture cap(inputVideoFile);

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

        cv::Mat H;
        if (std::filesystem::exists("H.xml"))
        {
            FileStorage fs;
            fs.open("H.xml", FileStorage::READ);
            fs["H"] >> H;
        }
        else
        {
            H = computeHomography(m_images[0], m_images[1]);
            cv::FileStorage file("H.xml", cv::FileStorage::WRITE);
            file << "H" << H;
        }

        auto h = H.inv();

        cv::Mat result;

        cv::warpPerspective(m_images[1], result, h, Size(1080 * 2, 1920));

        m_images[0].copyTo(result(Rect(0, 0, 1080, 1920)));
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

        imwrite("D:\\GITHUB\\stitchcv\\thirdparty\\SimplePanoStitcher\\sample\\00" + std::to_string(i) + ".png",
                m_images[i]);
    }
}

cv::Mat Stitcher::computeHomography(const cv::Mat & leftImage, const cv::Mat & rightImage)
{
    cv::Mat grayPartLeft, grayPartRight;

    const auto horizontalOverlap = static_cast<int>(ceil((1080.f * 4 - 3648) / 3.f)); // ~20% overlap

    const auto height = leftImage.size[0];
    const auto width = leftImage.size[1];

    if (m_leftImageMask.empty())
    {
        m_leftImageMask = cv::Mat::zeros(leftImage.size(), CV_8U);
        m_leftImageMask(cv::Rect(width - horizontalOverlap - 1, 0, horizontalOverlap, height)) = 1;
    }

    if (m_rightImageMask.empty())
    {
        m_rightImageMask = cv::Mat::zeros(leftImage.size(), CV_8U);
        m_rightImageMask(cv::Rect(0, 0, horizontalOverlap, height)) = 1;
    }

    cv::cvtColor(leftImage, grayPartLeft, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImage, grayPartRight, cv::COLOR_BGR2GRAY);

    std::vector<KeyPoint> keyPointsLeft, keyPointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    m_sift->detectAndCompute(grayPartLeft, m_leftImageMask, keyPointsLeft, descriptorsLeft);
    m_sift->detectAndCompute(grayPartRight, m_rightImageMask, keyPointsRight, descriptorsRight);

    const auto matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<std::vector<DMatch>> knnMatches;
    matcher->knnMatch(descriptorsLeft, descriptorsRight, knnMatches, 2);
    //-- Filter matches using the Lowe's ratio test
    const auto ratioThresh = 0.75f;
    std::vector<DMatch> goodMatches;
    for (const auto & knnMatch : knnMatches)
    {
        if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance)
        {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    std::vector<Point2f> leftPoints;
    std::vector<Point2f> rightPoints;
    for (const auto & goodMatch : goodMatches)
    {
        //-- Get the key points from the good matches
        leftPoints.push_back(keyPointsLeft[goodMatch.queryIdx].pt);
        rightPoints.push_back(keyPointsRight[goodMatch.trainIdx].pt);
    }

    // Homgraphic transformation from Left to Right image
    Mat H = cv::findHomography(leftPoints, rightPoints, RANSAC, 4.0);

    return H;
}
