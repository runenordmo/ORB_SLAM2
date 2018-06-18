#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <string>
#include <math.h> //atan2
#include <iomanip> // std::setw	
#include "opencv2/calib3d/calib3d.hpp"

//include our own header files
#include "CNNextractor.h"
#include "ORBextractor.h"

//Define pi
const double MY_PI = 3.14159265358979323846;

//Forward declarations
std::vector<cv::Mat> loadKittiGroundtruthPoses(std::string filename_gt);
std::string ZeroPadNumber(int num);
double rotationError(const cv::Mat T_error);
double translationError(const cv::Mat T_error);
cv::Mat toTransform(const cv::Mat & R, const cv::Mat & t);
bool findPose(const int firstFrameNum, const int secondFrameNum,
	const int firstCamNum, const int secondCamNum,
	const int kittiSeqNum,
	std::unique_ptr<ORB_SLAM2::ORBextractor> & mpORBextractorLeft,
	std::unique_ptr<ORB_SLAM2::ORBextractor> & mpORBextractorRight,
	ORB_SLAM2::CNNextractor & ext1, ORB_SLAM2::CNNextractor & ext2,
	const bool useLearnedFeatures,
	uint & numMatches, uint & numInliersAfterEssentialMat, uint & numInliersAfterRecoverPose,
	const cv::Mat K,
	cv::Mat & T_c2_c1);
double angle(std::vector<double> u, std::vector<double> v);

int main()
{
	//Calculating essential matrix to evaluate feature matching	
	//KITTI calibration matrix and baseline
	cv::Mat K_Kitti04to12 = (cv::Mat_<double>(3, 3) <<
		707.0912, 0.0, 601.8873,
		0.0, 707.0912, 183.1104,
		0.0, 0.0, 1.0);
	cv::Mat K_Kitti03 = (cv::Mat_<double>(3, 3) <<
		721.5377, 0.0, 609.5593,
		0.0, 721.5377, 172.854,
		0.0, 0.0, 1.0);
	cv::Mat K_Kitti00 = (cv::Mat_<double>(3, 3) <<
		718.856, 0.0, 607.1928,
		0.0, 718.856, 185.2157,
		0.0, 0.0, 1.0);

	const double baseline_Kitti = 0.5371506;
	const int HIGHEST_FRAME_NO_KITTI00 = 4540;
	const int HIGHEST_FRAME_NO_KITTI03 = 800;
	const int HIGHEST_FRAME_NO_KITTI04 = 270;
	const int HIGHEST_FRAME_NO_KITTI05 = 2760;
	//Change the two vars below according to seq. number
	int kittiSeqNum = 0;
	const int endFrameNum = HIGHEST_FRAME_NO_KITTI00 - 2;
	//

	const int startFrameNum = 0;
	int firstCamNum = 0, secondCamNum = 1; //stereo
	cv::Mat K;
	if (kittiSeqNum == 0) {
		K = K_Kitti00;
	}
	else if (kittiSeqNum == 3) {
		K = K_Kitti03;
	}
	else if (kittiSeqNum == 4) {
		K = K_Kitti04to12;
	}
	else if (kittiSeqNum == 5) {
		K = K_Kitti04to12;
	}
	
	bool useLearnedFeatures = false;

	//Load Kitti ground truth poses
	std::vector<cv::Mat> vTwcs = loadKittiGroundtruthPoses("KITTI0" + std::to_string(kittiSeqNum) + "_groundtruth.txt");
	//Descriptor files
	std::string descriptorFileName1 = "kitti_0" + std::to_string(kittiSeqNum) + "_cam" + std::to_string(firstCamNum) + "_descriptors.dat";
	std::string descriptorFileName2 = "kitti_0" + std::to_string(kittiSeqNum) + "_cam" + std::to_string(secondCamNum) + "_descriptors.dat";
	//CNN feature extractor
	ORB_SLAM2::CNNextractor ext1(2000, 4, 1, descriptorFileName1);
	ORB_SLAM2::CNNextractor ext2(2000, 4, 1, descriptorFileName2);
	
	
	//Define constants for ORB feature extraction
	const int ORB_nFeatures = 2000;
	const float ORB_scaleFactor = 1.2;
	const int ORB_nLevels = 8;
	const float ORB_iniThFast = 20;
	const float ORB_minThFast = 7;
	/*
	cv::Ptr<cv::ORB> orb_ext = cv::ORB::create();
	orb_ext->setFastThreshold(1);
	orb_ext->setMaxFeatures(2000);
	*/
	//ORB feature extractor
	std::unique_ptr<ORB_SLAM2::ORBextractor> pORBextractorLeft(new ORB_SLAM2::ORBextractor(ORB_nFeatures, ORB_scaleFactor, ORB_nLevels, ORB_iniThFast, ORB_minThFast));
	std::unique_ptr<ORB_SLAM2::ORBextractor> pORBextractorRight(new ORB_SLAM2::ORBextractor(ORB_nFeatures, ORB_scaleFactor, ORB_nLevels, ORB_iniThFast, ORB_minThFast));

	//
	std::array<uint, endFrameNum - startFrameNum + 1> vFirstFrameNumsOrb, vNumMatchessOrb, numInliersAfterEssentialMatsOrb, numInliersAfterRecoverPosesOrb;
	std::array<double, endFrameNum - startFrameNum + 1> rotationErrorOrb, translationErrorOrb;
	std::array<uint, endFrameNum - startFrameNum + 1> vFirstFrameNumsCnn, vNumMatchessCnn, numInliersAfterEssentialMatsCnn, numInliersAfterRecoverPosesCnn;
	std::array<double, endFrameNum - startFrameNum + 1> rotationErrorCnn, translationErrorCnn;

	//
	uint numMatches, numInliersAfterEssentialMat, numInliersAfterRecoverPose = 0;
	
	for (int currFrameNum = startFrameNum; currFrameNum <= endFrameNum; currFrameNum++) {
		int firstFrameNum = currFrameNum; //holding firstFrame at startFrame, !currFrameNum!
		int secondFrameNum = currFrameNum + 2; //secondFrame start 2 in front, then get further away

		cv::Mat T_c2_c1_orb;
		useLearnedFeatures = false;
		findPose(firstFrameNum, secondFrameNum, firstCamNum, secondCamNum, kittiSeqNum, pORBextractorLeft, pORBextractorRight, ext1, ext2, useLearnedFeatures, numMatches, numInliersAfterEssentialMat, numInliersAfterRecoverPose, K, T_c2_c1_orb);
		vFirstFrameNumsOrb[currFrameNum - startFrameNum] = currFrameNum; // !currFrameNum!
		vNumMatchessOrb[currFrameNum - startFrameNum] = numMatches;
		numInliersAfterEssentialMatsOrb[currFrameNum - startFrameNum] = numInliersAfterEssentialMat;
		numInliersAfterRecoverPosesOrb[currFrameNum - startFrameNum] = numInliersAfterRecoverPose;

		
		cv::Mat T_c2_c1_cnn;
		useLearnedFeatures = true;
		findPose(firstFrameNum, secondFrameNum, firstCamNum, secondCamNum, kittiSeqNum, pORBextractorLeft, pORBextractorRight, ext1, ext2, useLearnedFeatures, numMatches, numInliersAfterEssentialMat, numInliersAfterRecoverPose, K, T_c2_c1_cnn);
		vFirstFrameNumsCnn[currFrameNum - startFrameNum] = currFrameNum; // !currFrameNum!
		vNumMatchessCnn[currFrameNum - startFrameNum] = numMatches;
		numInliersAfterEssentialMatsCnn[currFrameNum - startFrameNum] = numInliersAfterEssentialMat;
		numInliersAfterRecoverPosesCnn[currFrameNum - startFrameNum] = numInliersAfterRecoverPose;
		

		cv::Mat T_gt2_gt1 = vTwcs[secondFrameNum].inv() * vTwcs[firstFrameNum]; //transform between frame2 and frame1
		if (firstCamNum == 0 && secondCamNum == 1) {
			T_gt2_gt1.at<double>(0, 3) -= baseline_Kitti;
		}

		std::vector<double> t_gt2_gt1 = { T_gt2_gt1.at<double>(0, 3),T_gt2_gt1.at<double>(1, 3),T_gt2_gt1.at<double>(2, 3) };

		//Calculate the error metrics for orb
			//Calculate rotation error for the essential matrix calculation
			//Calculate angle between ground truth translation direction and estimated translation direction 
		cv::Mat T_error_orb = T_c2_c1_orb.inv() * T_gt2_gt1;
		rotationErrorOrb[currFrameNum - startFrameNum] = rotationError(T_error_orb);
		
		std::vector<double> t_c2_c1_orb = { T_c2_c1_orb.at<double>(0, 3),T_c2_c1_orb.at<double>(1, 3),T_c2_c1_orb.at<double>(2, 3) };
		double angTransOrb = angle(t_gt2_gt1, t_c2_c1_orb);
		translationErrorOrb[currFrameNum - startFrameNum] = angTransOrb;

		//Calculate the error metrics for cnn
		cv::Mat T_error_cnn = T_c2_c1_cnn.inv() * T_gt2_gt1;
		rotationErrorCnn[currFrameNum - startFrameNum] = rotationError(T_error_cnn);

		std::vector<double> t_c2_c1_cnn = { T_c2_c1_cnn.at<double>(0, 3),T_c2_c1_cnn.at<double>(1, 3),T_c2_c1_cnn.at<double>(2, 3) };
		double angTransCnn = angle(t_gt2_gt1, t_c2_c1_cnn);
		translationErrorCnn[currFrameNum - startFrameNum] = angTransCnn;

		//Print to see how far we have come
		if (currFrameNum % 10 == 0) {
			std::cout << "Done finding errors for frame" << currFrameNum << std::endl; //!currFrameNum!
		}
	}
		
	//Send ORB matching data to text file, to plot in MATLAB
	std::ofstream myfileOrb("matchingDataOrb0" + std::to_string(kittiSeqNum) + "_diffTransMetric.txt");
	if (myfileOrb.is_open())
	{
		for (int firstFrameNum = startFrameNum; firstFrameNum <= endFrameNum; firstFrameNum++) {
			int idx = firstFrameNum - startFrameNum;
			myfileOrb << vFirstFrameNumsOrb[idx] << " "
				<< vNumMatchessOrb[idx] << " "
				<< numInliersAfterEssentialMatsOrb[idx] << " "
				<< numInliersAfterRecoverPosesOrb[idx] << " "
				<< rotationErrorOrb[idx] << " "
				<< translationErrorOrb[idx] << "\n";
		}
		myfileOrb.close();
	}
	else { std::cout << "Unable to open file"; return -1; }

	
	//Send CNN matching data to text file, to plot in MATLAB
	std::ofstream myfileCnn("matchingDataCnn0" + std::to_string(kittiSeqNum) + "_diffTransMetric.txt");
	if (myfileCnn.is_open())
	{
		for (int firstFrameNum = startFrameNum; firstFrameNum <= endFrameNum; firstFrameNum++) {
			int idx = firstFrameNum - startFrameNum;
			myfileCnn << vFirstFrameNumsCnn[idx] << " "
				<< vNumMatchessCnn[idx] << " "
				<< numInliersAfterEssentialMatsCnn[idx] << " "
				<< numInliersAfterRecoverPosesCnn[idx] << " "
				<< rotationErrorCnn[idx] << " "
				<< translationErrorCnn[idx] << "\n";
		}
		myfileCnn.close();
	}
	else { std::cout << "Unable to open file"; return -1; }
	

	return 0;
}

bool findPose(const int firstFrameNum, const int secondFrameNum,
	const int firstCamNum, const int secondCamNum,
	const int kittiSeqNum,
	std::unique_ptr<ORB_SLAM2::ORBextractor> & pORBextractorLeft,
	std::unique_ptr<ORB_SLAM2::ORBextractor> & pORBextractorRight,
	ORB_SLAM2::CNNextractor & ext1, ORB_SLAM2::CNNextractor & ext2,
	const bool useLearnedFeatures,
	uint & numMatches, uint & numInliersAfterEssentialMat, uint & numInliersAfterRecoverPose,
	const cv::Mat K,
	cv::Mat & T_c2_c1) {

	//Vectors of keypoints, and their corresponding descriptors
	std::vector<cv::KeyPoint> kpts_vec1, kpts_vec2;
	cv::Mat desc1, desc2;
	//
	std::vector<cv::Point2f> selected_points1, selected_points2;
	cv::Mat R, t;
	cv::Mat E;

	//Feature extractor
	//Read images
	std::string imageName1 = "kitti_0" + std::to_string(kittiSeqNum) + "/0" + std::to_string(kittiSeqNum) + "/image_" + std::to_string(firstCamNum) + "/" + ZeroPadNumber(firstFrameNum) + ".png";
	std::string imageName2 = "kitti_0" + std::to_string(kittiSeqNum) + "/0" + std::to_string(kittiSeqNum) + "/image_" + std::to_string(secondCamNum) + "/" + ZeroPadNumber(secondFrameNum) + ".png";
	cv::Mat image1 = cv::imread(imageName1, cv::IMREAD_GRAYSCALE);
	if (!image1.data) { std::cout << " Error reading " << image1 << std::endl; return 0; }
	cv::Mat image2 = cv::imread(imageName2, cv::IMREAD_GRAYSCALE);
	if (!image2.data) { std::cout << " Error reading " << image2 << std::endl; return 0; }

	//Brute-force matcher
	cv::BFMatcher matcher;
	bool useCrossCheckMatching = true;
	if (useLearnedFeatures) matcher = cv::BFMatcher(cv::NORM_L2, useCrossCheckMatching);
	else matcher = cv::BFMatcher(cv::NORM_HAMMING, useCrossCheckMatching);

	if (useLearnedFeatures) {
		ext1(image1, firstFrameNum, 0, kpts_vec1, desc1);
		ext2(image2, secondFrameNum, 0, kpts_vec2, desc2);
	}
	else {
		/*
		orb_ext->detectAndCompute(image1, cv::noArray(), kpts_vec1, desc1);
		orb_ext->detectAndCompute(image2, cv::noArray(), kpts_vec2, desc2);
		*/
		(*pORBextractorLeft)(image1, cv::Mat(), kpts_vec1, desc1);
		(*pORBextractorRight)(image2, cv::Mat(), kpts_vec2, desc2);
	}

	//Cross-ratio-test match
	std::vector<cv::DMatch> matches;
	matcher.match(desc1, desc2, matches);

	selected_points1.clear();
	selected_points2.clear();
	for (int i = 0; i < matches.size(); i++) {
		selected_points1.push_back(kpts_vec1[matches[i].queryIdx].pt);
		selected_points2.push_back(kpts_vec2[matches[i].trainIdx].pt);
	}

	//Save matches image
	/*
	cv::Mat src;
	cv::hconcat(image1, image2, src);
	for (int i = 0; i < selected_points1.size(); i++) {
	cv::line(src, selected_points1[i],
	cv::Point2f(selected_points2[i].x + image1.cols, selected_points2[i].y),
	1, 1, 0);
	}
	cv::imwrite(std::to_string(useLearnedFeatures)+"match-result"+".png", src);
	*/

	//Optional: Save the number of matches
	numMatches = selected_points1.size();

	//Find the essential matrix
	cv::Mat mask;
	E = cv::findEssentialMat(selected_points1, selected_points2, K, cv::RANSAC, 0.999, 1.0, mask);

	std::vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			inlier_match_points1.push_back(selected_points1[i]);
			inlier_match_points2.push_back(selected_points2[i]);
		}
	}
	//Save the number of inliers
	numInliersAfterEssentialMat = cv::sum(mask)[0];

	//Optional: Save the image with essential matrix-inliers
	/*
	cv::Mat src2;
	cv::hconcat(image1, image2, src2);
	for (int i = 0; i < inlier_match_points1.size(); i++) {
	cv::line(src2, inlier_match_points1[i],
	cv::Point2f(inlier_match_points2[i].x + image1.cols, inlier_match_points2[i].y),
	1, 1, 0);
	}
	cv::imwrite(std::to_string(useLearnedFeatures) + "inlier_match_points"  + ".png", src2);
	*/

	//Recover the pose from the essential matrix
	mask.release();
	cv::recoverPose(E, inlier_match_points1, inlier_match_points2, K, R, t, mask);

	T_c2_c1 = toTransform(R, t);

	//Count the number of inliers after recovering the pose
	numInliersAfterRecoverPose = 0;
	for (int i = 0; i < mask.rows; i++) {
		if (mask.at<unsigned char>(i)) {
			numInliersAfterRecoverPose++;
		}
	}
}

cv::Mat toTransform(const cv::Mat & R, const cv::Mat & t) {
	double myT_array[4 * 4] = {
		R.at<double>(0, 0),R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
		R.at<double>(1, 0),R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
		R.at<double>(2, 0),R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2),
		0,				  0,				 0,					1 };
	return cv::Mat(4, 4, CV_64F, myT_array).clone();
}

double rotationError(const cv::Mat T_error) {
	double a = T_error.at<double>(0, 0);
	double b = T_error.at<double>(1, 1);
	double c = T_error.at<double>(2, 2);
	double trace = a + b + c;
	double d = 0.5*(trace - 1.0);
	return acos(std::max(std::min(d, 1.0), -1.0)); //angle of angle-axis rep.
}

double translationError(const cv::Mat T_error) {
	double dx = T_error.at<double>(0, 3);
	double dy = T_error.at<double>(1, 3);
	double dz = T_error.at<double>(2, 3);
	return sqrt(dx * dx + dy * dy + dz * dz); //euclidean distance
}

double angle(std::vector<double> u, std::vector<double> v) {
	std::vector<double> cross = {
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0]
	};
	double eucNormOfCross = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
	double dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

	double angle = atan2(eucNormOfCross, dot);
	return angle;
}

std::vector<cv::Mat> loadKittiGroundtruthPoses(std::string filename_gt) {
	std::ifstream f;
	f.open(filename_gt.c_str());
	std::vector<cv::Mat> vTwcs;
	if (!f) {
		std::cout << "Could not open file!" << std::endl; return vTwcs;
	}
	while (!f.eof()) {
		cv::Mat Twc = cv::Mat::eye(4, 4, CV_64F);
		f >> Twc.at<double>(0, 0) >> Twc.at<double>(0, 1) >> Twc.at<double>(0, 2) >> Twc.at<double>(0, 3) >>
			Twc.at<double>(1, 0) >> Twc.at<double>(1, 1) >> Twc.at<double>(1, 2) >> Twc.at<double>(1, 3) >>
			Twc.at<double>(2, 0) >> Twc.at<double>(2, 1) >> Twc.at<double>(2, 2) >> Twc.at<double>(2, 3);
		vTwcs.push_back(Twc);
	}
	f.close();
}

std::string ZeroPadNumber(int num)
{
	std::ostringstream ss;
	ss << std::setw(6) << std::setfill('0') << num;
	std::string result = ss.str();
	if (result.length() > 6){ result.erase(0, result.length() - 6); }
	return result;
}