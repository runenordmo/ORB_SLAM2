//
// Created by rune on 06.05.18.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <iostream>
// s_cwd
#include <unistd.h>

#include "CNNextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
	const int EDGE_THRESHOLD = 19;
	const int CNN_DESCRIPTOR_SIZE = 256;

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
		//std::string s_cwd(getcwd(NULL,0));
		//mFilename = s_cwd + "/Vocabulary/kitti_04_descriptors.dat";
		//mFilename = s_cwd + "/Vocabulary/euroc_mh1_descriptors.dat";
		//mFilename = s_cwd + "/Vocabulary/tum_fr1_xyz_descriptors.dat";
		//mFilename = s_cwd + "/Vocabulary/tum_fr1_xyz_descriptors2.dat";
	}

	//CNN specific
	void CNNextractor::operator()(cv::InputArray _image, int frameNumber, cv::InputArray _mask,
		std::vector<cv::KeyPoint>& _keypoints, cv::Mat & _descriptors, const std::string &descriptorFile) {

		if (_image.empty())
			return;

		Mat image = _image.getMat();
		assert(image.type() == CV_8UC1);

		// Pre-compute the scale pyramid
		ComputePyramid(image);

		// CNN specific part
		//usleep(1000000);
		_keypoints.clear();
		_descriptors = cv::Mat();


		std::ifstream f{ descriptorFile, std::ios::binary };
		if (!f) {
			std::cout << "Can't open: " << descriptorFile << std::endl;
			return;
		}

		if (mFrameIndexesInFile.empty()) { //Must read through file to find all the frames' file indexes
			mFrameIndexesInFile.push_back(f.tellg());
			while (true) {
				//Read first two coordinates, abort if reading goes wrong
				cv::Point dummyPt;
				f.read(reinterpret_cast<char*>(&dummyPt.x), sizeof(std::int32_t));
				f.read(reinterpret_cast<char*>(&dummyPt.y), sizeof(std::int32_t));
				if (f.eof()) { //if we have reached EOF, the last index added didn't correspond to a frame
					if (!mFrameIndexesInFile.empty()) { mFrameIndexesInFile.pop_back(); }
					break;
				}
				if (!f) { 
					std::cout << "Error reading keyPoints for saving the frame's indexes in file" << std::endl;
					mFrameIndexesInFile.clear(); 
					_keypoints.clear(); 
					_descriptors = cv::Mat(); 
					f.close(); 
					return; 
				}

				bool last_feature_in_frame = (dummyPt.x == -1 && dummyPt.y == -1);
				if (!last_feature_in_frame) {
					//Read descriptor, abort if reading goes wrong
					std::array<float, CNN_DESCRIPTOR_SIZE> dummyDesc;
					f.read(reinterpret_cast<char *>(&dummyDesc), sizeof(dummyDesc));
					if (!f)
					{
						std::cout << "Error reading descriptors for saving the frame's indexes in file" << std::endl;
						mFrameIndexesInFile.clear();
						_keypoints.clear();
						_descriptors = cv::Mat();
						f.close();
						return;
					}
				}
				else {
					mFrameIndexesInFile.push_back(f.tellg());
				}
			}
			f.close();
		}

		//At this point, mFrameIndexesInFile is not empty
		//Read features from the appropriate frame
		std::ifstream f2{ descriptorFile, std::ios::binary };
		if (!f2) {
			std::cout << "Can't open: " << descriptorFile << std::endl;
			return;
		}

		if (frameNumber < mFrameIndexesInFile.size()) {
			f2.seekg(mFrameIndexesInFile[frameNumber]);
		}
		else {
			_keypoints.clear(); _descriptors = cv::Mat(); return;
		}

		while (true) {
			//Set up variables to save keypoint and descriptor in
			cv::KeyPoint keyP(0, 0, 1, -1, 0, 1, -1); //check the parameters once more?
			std::array<float, CNN_DESCRIPTOR_SIZE> desc;

			//Read first two coordinates, abort if reading goes wrong
			cv::Point pt;
			f2.read(reinterpret_cast<char*>(&pt.x), sizeof(std::int32_t));
			f2.read(reinterpret_cast<char*>(&pt.y), sizeof(std::int32_t));
			if (f2.eof())
			{
				_keypoints.clear();
				_descriptors = cv::Mat();
				f2.close();
				std::cout << "Error: end of file at frame: " << frameNumber << std::endl;
				return;
			}
			if (!f2)
			{
				_keypoints.clear();
				_descriptors = cv::Mat();
				f2.close();
				std::cout << "Error reading keyPoint at frame: " << frameNumber << std::endl;
				return;
			}

			bool last_feature_in_frame = (pt.x == -1 && pt.y == -1);
			if (!last_feature_in_frame) {
				//Read descriptor, abort if reading goes wrong
				f2.read(reinterpret_cast<char *>(&desc), sizeof(desc));
				if (!f2)
				{
					_keypoints.clear();
					_descriptors = cv::Mat();
					f2.close();
					std::cout << "Error reading descriptor at frame: " << frameNumber << std::endl;
					return;
				}

				//Save keypoint and descriptor
				keyP.pt = pt;
				_keypoints.push_back(keyP);
				_descriptors.push_back(cv::Mat(desc, true).reshape(1, 1));
			}
			else
			{ //ready to return the features!
				f2.close();
				return;
			}
		}
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