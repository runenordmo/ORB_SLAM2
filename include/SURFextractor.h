//
// Created by rune on 07.04.18.
//

#ifndef ORB_SLAM2_SURFEXTRACTOR_H
#define ORB_SLAM2_SURFEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{

class SURFextractor {
public:

    //ORBextractor(int nfeatures, float scaleFactor, int nlevels,
    //            int iniThFAST, int minThFAST);
    // NOT SURE HOW TO HANDLE THE SCALE WITH THE SURF EXTRACTOR
    SURFextractor();

    ~SURFextractor(){}

    // Compute the SURF features and descriptors on an image.
    // Mask is ignored in the current implementation.
    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

    int inline GetLevels(){
        //return nlevels;}
        return 1;}

    float inline GetScaleFactor(){
        //return scaleFactor;}
        return 1;}

    std::vector<float> inline GetScaleFactors(){
        //return mvScaleFactor;
        return std::vector<float>(8,1);
    }

    std::vector<float> inline GetInverseScaleFactors(){
        //return mvInvScaleFactor;
        return std::vector<float>(8,1);
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        //return mvLevelSigma2;
        return std::vector<float>(8,1);
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        //return mvInvLevelSigma2;
        return std::vector<float>(8,1);
    }
protected:

    // USE OCTTREE HERE?

    int nfeatures;

    cv::Ptr<cv::Feature2D> mSurfDetector;

    // USE SCALEFACTOR AND NLEVELS HERE?

};



} //namespace ORB_SLAM


#endif //ORB_SLAM2_SURFEXTRACTOR_H
