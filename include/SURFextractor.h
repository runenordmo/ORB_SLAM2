//
// Created by rune on 07.04.18.
//

#ifndef SURFEXTRACTOR_H
#define SURFEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM2
{

class SURFextractor {
public:

    SURFextractor(int nfeatures, float scaleFactor, int nlevels);

    ~SURFextractor(){}

    // Compute the SURF features and descriptors on an image.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);

    int nfeatures;
    double scaleFactor;
    int nlevels;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    cv::Ptr<cv::Feature2D> mSurfDetector;

    // USE SCALEFACTOR AND NLEVELS HERE?

};



} //namespace ORB_SLAM


#endif //SURFEXTRACTOR_H
