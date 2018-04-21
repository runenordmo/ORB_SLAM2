//
// Created by rune on 07.04.18.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "SURFextractor.h"

// SURF SPECIFIC:

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

SURFextractor::SURFextractor() {
    double hessian = 1000.0;
    mSurfDetector = xfeatures2d::SURF::create(hessian);
}

void SURFextractor::operator()(const cv::_InputArray &image, const cv::_InputArray &mask,
                                          std::vector<cv::KeyPoint> &keypoints, const cv::_OutputArray &descriptors) {
    mSurfDetector->detectAndCompute(image, mask, keypoints, descriptors, false);
}

} //namespace ORB_SLAM