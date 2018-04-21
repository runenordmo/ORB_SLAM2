//
// Created by rune on 08.04.18.
//

#ifndef ORB_SLAM2_SURFMATCHER_H
#define ORB_SLAM2_SURFMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

namespace ORB_SLAM2{

class SURFmatcher {

    // Computes the euclidean distance between two SURF descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

};

} //namespace ORB_SLAM

#endif //ORB_SLAM2_SURFMATCHER_H
