//
// Created by rune on 08.04.18.
//

#include "SURFmatcher.h"

using namespace cv;

namespace ORB_SLAM2{

int SURFmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    return norm(a,b,NORM_L2);
}


} //namespace ORB_SLAM
