//
// Created by rune on 06.05.18.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
// s_cwd
#include <unistd.h>

#include "CNNextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
const int EDGE_THRESHOLD = 19;

CNNextractor::CNNextractor(int _nfeatures, float _scaleFactor, int _nlevels) :
	nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels)
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
	std::string s_cwd(getcwd(NULL,0));
  	mFilename = s_cwd + "/Vocabulary/kitti_04_descriptors.dat";
}

//CNN specific
void CNNextractor::operator()(cv::InputArray _image, int frameNumber, cv::InputArray _mask,
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
	
	// Keep track of frame number, keypoints and descriptors
	int currFrameNumber = 0;
	//f.seekg(0);

	while (!f.eof()) {

		std::array<float, 512> desc; //single descriptor
		cv::KeyPoint keyP(0, 0, 1, -1, 0, 0, -1); //single keypoint

		cv::Point pt;
		f.read(reinterpret_cast<char*>(&pt.x), sizeof(std::int32_t));
		f.read(reinterpret_cast<char*>(&pt.y), sizeof(std::int32_t));

		if (!f) { _keypoints.clear(); _descriptors = cv::Mat(); f.close(); return; }
		else {
			bool last_feature_in_frame = (pt.x == -1 && pt.y == -1);
			if (!last_feature_in_frame) {
				//read descriptor
				f.read(reinterpret_cast<char *>(&desc), sizeof(desc));
				if (!f) { _keypoints.clear();  _descriptors = cv::Mat(); f.close(); return; }
				else { //reading went well, save keypoint and descriptor
					if (currFrameNumber == frameNumber) { //ready to return the features!
						keyP.pt = pt;
						_keypoints.push_back(keyP);
						_descriptors.push_back(cv::Mat(desc, true).reshape(1, 1));
					}
				}
			}
			else { //last feature in frame
				if (currFrameNumber == frameNumber) { //ready to return the features!
					f.close(); return;
				}
				else {
					currFrameNumber++;
				}
			}
		}	
	}
	//reached EOF
	_keypoints.clear();  _descriptors = cv::Mat(); f.close(); return;
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