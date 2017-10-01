/*
 * File Description :   File contains DLIB model class object for DLIB face recognition.
 * File Name        :   dlib_model.hpp
 * Author           :   Srini Ananthakrishnan
 * Date             :   09/23/2017
 * Github           :   https://github.com/srianant
 * Reference        :   DLIB c++ examples (https://github.com/davisking/dlib)
 */

#ifndef DLIB_MODEL_H
#define DLIB_MODEL_H

#include <iostream>
#include <vector>

// 3rdpary depencencies
#include "dlib/opencv.h"
#include "dlib/image_processing/full_object_detection.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/dnn.h"
#include "dlib/string.h"
#include "dlib/clustering.h"
#include "dlib/svm_threaded.h"

#include "openpose/user_code/pose_model.hpp"

using namespace std;
using namespace dlib;
using namespace cv;

// ----------------------------------------------------------------------------------------
// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
// Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

// We need a face detector.  We will use this to get bounding boxes for
// each face in an image.
frontal_face_detector detector = get_frontal_face_detector();
// And we also need a shape_predictor.  This is the tool that will predict face
// landmark positions given an image and face bounding box.  Here we are just
// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
// as a command line argument.
shape_predictor sp;
// And finally we load the DNN responsible for face recognition.
anet_type net;

std::vector<matrix<rgb_pixel>> train_faces;
std::vector<matrix<float,0,1>> train_face_descriptors;
std::vector <string> dlib_face_names;

// DLIB Face recognition model
class OP_API dlib_model
{
	public:
        void readTrainFilenames( std::string& filename, std::string& dirName, std::vector<string>& trainFilenames);

        std::string readStdIn(int timeout);

        int facerec_dlib_train_face_images(int initialize_done);

        void dlib_facerec_from_faceRectangles(cv::Mat &_outputImage,const Array<float>& _faceKeypoints,
                                                int & _frame_count_per_sample, bool *_face_capture, bool _debug);

	private:
};


#endif // DLIB_MODEL_H

