/*
 * File Description :   File contains OPENPOSE model class object for emotion, gesture and pose recognition.
 * File Name        :   pose_model.hpp
 * Author           :   Srini Ananthakrishnan
 * Date             :   09/23/2017
 * Github           :   https://github.com/srianant
 * Reference        :   OPENPOSE c++ examples (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
 */

#ifndef POSE_MODEL_H
#define POSE_MODEL_H

#include <vector>
#include <utility>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <tuple>
#include <unistd.h>
#include <poll.h>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/gpu/gpu.hpp"
#endif  // USE_OPENCV
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <openpose/face/faceDetector.hpp>
#include <openpose/face/faceExtractor.hpp>
#include <openpose/face/faceParameters.hpp>
#include <openpose/hand/handDetector.hpp>
#include <openpose/hand/handExtractor.hpp>
#include <openpose/hand/handRenderer.hpp>
#include <openpose/core/renderer.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/poseParameters.hpp>
// Keras dependencies
#include "openpose/user_code/kerasify_model.hpp"

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/vector.hpp"

using namespace boost::archive;
using namespace std;
using namespace cv;
using namespace op;

enum keypoints_types { KP_POSE, KP_LEFT_HAND, KP_RIGHT_HAND, KP_FACE, MAX_KEYPOINTS };

// Timesteps per samples
const int timesteps = 5; // 5 frames per sample

// 2-dim data to store different postures
const int max_pose_count = 10;
const int max_pose_score_pair = 36;  // 34 pairs for pose + 2 padding for multiples of 4
typedef std::array<string, max_pose_count> pose;
typedef std::array<double,max_pose_score_pair*timesteps> pose_sample_type;


// 2-dim data to store different hand actions
const int max_hand_count = 10;
const int max_hand_score_pair = 40;  // 40 pairs for hand
typedef std::array<string,max_hand_count> hand;
typedef std::array<double,max_hand_score_pair*timesteps> hand_sample_type;

// 2-dim data to store different faces
const int max_face_count = 10;
const int max_face_score_pair = 96; // 96 pairs (94 + 2 pairs for padding for multiples of 4)
typedef std::array<string,max_face_count> face;
typedef std::array<double,max_face_score_pair*timesteps> face_sample_type;

typedef std::array<int,max_pose_count> label_array;

class OP_API pose_model
{
    public:

        void save_to_file(string filename, hand &_hand_names);

        hand load_from_file(string filename);

        float computeDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB, bool cosine);

        std::tuple<int,float> get_pose_predictions(int keypoints_size, pose_sample_type &_pose_sample, KerasModel &_k_model);

        std::tuple<int,float> get_hand_predictions(int keypoints_size, hand_sample_type &_hand_sample, KerasModel &_k_model);

        std::tuple<int,float> get_face_predictions(int keypoints_size, face_sample_type &_face_sample, KerasModel &_k_model);

        void getSamplesFromPoseKeypoints(cv::Mat &_outputImage, pose_sample_type &pose_sample, const Array<float>& keypoints,
                                     label_array &_label_count, const std::vector<unsigned int>& pairs, const int label,
                                     const float threshold, pose &class_name, int & _frame_count_per_sample, int & samples_count,
                                     pose & _predicted_class_str, pose & _predicted_score_str, const float _classifier_threshold,
                                     std::vector<double>& _pose_labels, std::vector<pose_sample_type>& _pose_samples,
                                     bool _generate, bool _distance, bool debug, KerasModel &_k_model);

        void getSamplesFromHandKeypoints(cv::Mat &_outputImage, hand_sample_type &hand_sample, const Array<float>& keypoints,
                                     label_array &_label_count, const std::vector<unsigned int>& pairs, const int label,
                                     const float threshold, hand &class_name, int & _frame_count_per_sample, int & samples_count,
                                     hand & _predicted_class_str, hand & _predicted_score_str, const float _classifier_threshold,
                                     std::vector<double>& _hand_labels, std::vector<hand_sample_type>& _hand_samples,
                                     keypoints_types type, bool _generate, bool _distance, bool debug, KerasModel &_k_model);

        void getSamplesFromFaceKeypoints(cv::Mat &_outputImage, face_sample_type &face_sample, const Array<float>& keypoints,
                                     label_array &_label_count, const std::vector<unsigned int>& pairs, const int _label,
                                     const float threshold, face &class_name, int & _frame_count_per_sample, int & samples_count,
                                     face & _predicted_class_str, face & _predicted_score_str, const float _classifier_threshold,
                                     std::vector<double>& _face_labels, std::vector<face_sample_type>& _face_samples,
                                     bool _generate, bool _distance, bool debug, KerasModel &_k_model);

    private:
};

#endif // POSE_MODEL_H
