#ifndef CNNEXTRACTOR_H
#define CNNEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM2
{

	class CNNextractor {
	public:

		//CNN specific
		CNNextractor(int nfeatures, float scaleFactor, int nlevels);

		~CNNextractor() {} 

		// Compute the CNN features and descriptors on an image.
		// Mask is ignored in the current implementation.
		//CNN specific
		void operator()(cv::InputArray _image, int frameNumber, cv::InputArray _mask,
			std::vector<cv::KeyPoint>& _keypoints, cv::Mat & _descriptors, const std::string &descriptorFile);

		int inline GetLevels() {
			return nlevels;
		}

		float inline GetScaleFactor() {
			return scaleFactor;
		}

		std::vector<float> inline GetScaleFactors() {
			return mvScaleFactor;
		}

		std::vector<float> inline GetInverseScaleFactors() {
			return mvInvScaleFactor;
		}

		std::vector<float> inline GetScaleSigmaSquares() {
			return mvLevelSigma2;
		}

		std::vector<float> inline GetInverseScaleSigmaSquares() {
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

		//CNN specific
		std::string mFilename = "../Vocabulary/kitti_04_descriptors.dat";
		std::vector<uint64_t> mFrameIndexesInFile;
	};

} //namespace ORB_SLAM


#endif //CNNEXTRACTOR_H