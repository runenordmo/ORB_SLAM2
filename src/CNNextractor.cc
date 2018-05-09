//
// Created by rune on 06.05.18.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>

#include "CNNextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
const int EDGE_THRESHOLD = 19;

CNNextractor::CNNextractor(int _nfeatures, float _scaleFactor, int _nlevels,
	std::string filename) :
	nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
	mFilename(filename), mFileInputPosition(0)
{
	mvScaleFactor.resize(nlevels);
	mvLevelSigma2.resize(nlevels);
	mvScaleFactor[0] = 1.0f;
	mvLevelSigma2[0] = 1.0f;
	for (int i = 1; i<nlevels; i++)
	{
		mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
		mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
	}

	mvInvScaleFactor.resize(nlevels);
	mvInvLevelSigma2.resize(nlevels);
	for (int i = 0; i<nlevels; i++)
	{
		mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
		mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
	}

	mvImagePyramid.resize(nlevels);
}

void CNNextractor::operator()(cv::InputArray _image, cv::InputArray _mask,
	std::vector<cv::KeyPoint>& _keypoints, cv::Mat & _descriptors){

	if (_image.empty())
		return;

	Mat image = _image.getMat();
	assert(image.type() == CV_8UC1);

	// Pre-compute the scale pyramid
	ComputePyramid(image);

	// CNN specific part
	_keypoints.clear();
	_descriptors = cv::Mat();

	std::ifstream f{ mFilename, std::ios::binary };
	if (!f) { //couldn't open the file
		return;
	}

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	while (!f.eof()) {
		std::array<float, 512> desc;
		cv::KeyPoint keyP(0, 0, 1, -1, 0, 0, -1);
		f.seekg(mFileInputPosition);
		cv::Point pt;
		f.read(reinterpret_cast<char*>(&pt.x), sizeof(std::int32_t));
		f.read(reinterpret_cast<char*>(&pt.y), sizeof(std::int32_t));
		keyP.pt = pt;

		bool last_feature_in_frame = keyP.pt.x == -1 || keyP.pt.y == -1;
		if (last_feature_in_frame) {
			mFileInputPosition += 2 * sizeof(std::int32_t);
			if (f) { //returning the features
				_descriptors = descriptors;
				_keypoints = keypoints;
				return;
			}
			else { //something went wrong while reading the file
				return;
			}
		}

		f.read(reinterpret_cast<char *>(&desc), sizeof(desc));
		// Add read keypoints and corresponding descriptors
		keypoints.push_back(keyP);
		descriptors.push_back(cv::Mat(desc, true).reshape(1, 1));
		mFileInputPosition += 2 * sizeof(std::int32_t) + sizeof(desc);
	}

	//reached EOF
}

void CNNextractor::ComputePyramid(cv::Mat image)
{

	for (int level = 0; level < nlevels; ++level)
	{
		float scale = mvInvScaleFactor[level];
		Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
		Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
		Mat temp(wholeSize, image.type()), masktemp;
		mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

		// Compute the resized image
		if (level != 0)
		{
			resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

			copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
				BORDER_REFLECT_101 + BORDER_ISOLATED);
		}
		else
		{
			copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
				BORDER_REFLECT_101);
		}
	}

}

} //namespace ORB_SLAM