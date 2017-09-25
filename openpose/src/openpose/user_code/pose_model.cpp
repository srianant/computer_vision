/*
 * File Description :   File defines OPENPOSE model class object functions for emotion, gesture and pose recognition.
 * File Name        :   pose_model.cpp
 * Author           :   Srini Ananthakrishnan
 * Date             :   09/23/2017
 * Github           :   https://github.com/srianant
 * Reference        :   OPENPOSE c++ examples (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
 */

#include <openpose/user_code/pose_model.hpp>

// Compute L2 or Cosine distance
float pose_model::computeDistance(const Array<float>& keypoints, const int person, const int elementA, const int elementB, bool _l2_distance)
{
    try
    {
        const auto keypointPtr = keypoints.getConstPtr() + person * keypoints.getSize(1) * keypoints.getSize(2);

        const auto x = keypointPtr[elementA*3];
        const auto y = keypointPtr[elementA*3+1];
        const auto a = keypointPtr[elementB*3];
        const auto b = keypointPtr[elementB*3+1];

        if (_l2_distance) // Euclidean Distance
            return (std::sqrt((x-a)*(x-a) + (y-b)*(y-b)));
        else // Cosine Distance
            return 1 - ((a*x + b*y)/((std::sqrt(std::abs(a*a)+std::abs(b*b)))*(std::sqrt(std::abs(x*x)+std::abs(y*y)))));
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1.f;
    }
}

// Get pose predictions from kerasify classifier model
std::tuple<int,float> pose_model::get_pose_predictions(int keypoints_size, pose_sample_type &_pose_sample, KerasModel &_k_model)
{
    if(_k_model.num_layers < 1)
        return std::make_tuple(0, 0.0);

    const int i = 1;                // single test sample
    const int j = timesteps;        // timesteps in single sample
    const int k = keypoints_size;   // per keypoint vector * timesteps

    // Input 3D tensor object class
    Tensor in(i, j, k);

    // Assign tensor object with input test sample
    for (int ii = 0; ii < i; ii++) {
        for (int jj = 0; jj < j; jj++) {
            for (int kk = 0; kk < k; kk++) {
                in(ii, jj, kk) = _pose_sample[(jj*k)+kk];
            }
        }
    }

    // Output tensor object class
    Tensor out;

    // Run prediction. Equivalent to keras model.predict(X_test).
    // Applies activations and per keras layer dot product of weights/biases for input test sample.
    _k_model.Apply(&in, &out);

    // Check if its one hot encoded, multi-label softmax
    if ((out.size() == 1) && ((out.get_1D_shape() > 1))) {

        //out.Print();

        float arr[out.get_1D_shape()];
        out.get_data(&arr[0]);

        int label = 0;
        float score = arr[0];

        for (auto i = 1; i < out.get_1D_shape(); i++)
        {
            if (arr[i] > score)
            {
                score = arr[i];
                label = i;
            }
        }
        return std::make_tuple(label, score);
    }
    else // sigmoid, softplus, etc.
    {
        float l = out(0);
        return std::make_tuple(round(l), 0);
    }
}

// Get hand predictions from kerasify classifier model
std::tuple<int,float> pose_model::get_hand_predictions(int keypoints_size, hand_sample_type &_hand_sample, KerasModel &_k_model)
{
    if(_k_model.num_layers < 1)
        return std::make_tuple(0, 0.0);

    const int i = 1;                // single test sample
    const int j = timesteps;        // timesteps in single sample
    const int k = keypoints_size;   // per keypoint vector * timesteps

    // Input 3D tensor object class
    Tensor in(i, j, k);

    // Assign tensor object with input test sample
    for (int ii = 0; ii < i; ii++) {
        for (int jj = 0; jj < j; jj++) {
            for (int kk = 0; kk < k; kk++) {
                in(ii, jj, kk) = _hand_sample[(jj*k)+kk];
            }
        }
    }

    // Output tensor object class
    Tensor out;

    // Run prediction. Equivalent to keras model.predict(X_test).
    // Applies activations and per keras layer dot product of weights/biases for input test sample.
    _k_model.Apply(&in, &out);

    // Check if its one hot encoded, multi-label softmax
    if ((out.size() == 1) && ((out.get_1D_shape() > 1))) {

        //out.Print();

        float arr[out.get_1D_shape()];
        out.get_data(&arr[0]);

        int label = 0;
        float score = arr[0];

        for (auto i = 1; i < out.get_1D_shape(); i++)
        {
            if (arr[i] > score)
            {
                score = arr[i];
                label = i;
            }
        }
        return std::make_tuple(label, score);
    }
    else // sigmoid, softplus, etc.
    {
        float l = out(0);
        return std::make_tuple(round(l), 0);
    }
}

// Get face predictions from kerasify classifier model
std::tuple<int,float> pose_model::get_face_predictions(int keypoints_size, face_sample_type &_face_sample, KerasModel &_k_model)
{
    if(_k_model.num_layers < 1)
        return std::make_tuple(0, 0.0);

    const int i = 1;                // single test sample
    const int j = timesteps;        // timesteps in single sample
    const int k = keypoints_size;   // per keypoint vector * timesteps

    // Input 3D tensor object class
    Tensor in(i, j, k);

    // Assign tensor object with input test sample
    for (int ii = 0; ii < i; ii++) {
        for (int jj = 0; jj < j; jj++) {
            for (int kk = 0; kk < k; kk++) {
                in(ii, jj, kk) = _face_sample[(jj*k)+kk];
            }
        }
    }

    // Output tensor object class
    Tensor out;

    // Run prediction. Equivalent to keras model.predict(X_test).
    // Applies activations and per keras layer dot product of weights/biases for input test sample.
    _k_model.Apply(&in, &out);

    // Check if its one hot encoded, multi-label softmax
    if ((out.size() == 1) && ((out.get_1D_shape() > 1))) {

        //out.Print();

        float arr[out.get_1D_shape()];
        out.get_data(&arr[0]);

        int label = 0;
        float score = arr[0];

        // find label with high probability score
        for (auto i = 1; i < out.get_1D_shape(); i++)
        {
            if (arr[i] > score)
            {
                score = arr[i];
                label = i;
            }
        }
        // return label and score with highest score
        return std::make_tuple(label, score);
    }
    else // sigmoid, softplus, etc.
    {
        float l = out(0);
        return std::make_tuple(round(l), 0);
    }
}

// -------------------------------------------------------------
// Functions to obtain samples (pose, hand, face) from Keypoints
// -------------------------------------------------------------
// Useful meanings:
// ----------------
// keypoints    : a vector of fixed dimension. (keypoint type: pose = 36, hand = 40, face = 96)
// timesteps    : number of image frames consider for one sample
// sample       : N keypoints at different timestep (eg. keypoints of 5 frames)
// label        : class number assigned to each sample
// samples      : vector of multiple sample(s)

// Get samples from pose keypoints
void pose_model::getSamplesFromPoseKeypoints(cv::Mat &_outputImage, pose_sample_type &pose_sample, const Array<float>& keypoints,
                                 label_array &_label_count, const std::vector<unsigned int>& pairs, const int label,
                                 const float threshold, pose &class_name, int & _frame_count_per_sample, int & samples_count,
                                 pose & _predicted_class_str, pose & _predicted_score_str, const float _classifier_threshold,
                                 std::vector<double>& _pose_labels, std::vector<pose_sample_type>& _pose_samples,
                                 bool _generate, bool _distance, bool _debug, KerasModel &_k_model)
{
    const auto numberKeypoints = keypoints.getSize(1);  // keypoints vector size (dimension for one timestep)
    const auto thresholdRectangle = 0.1f;               // rectangle confidence threshold
    int _max_keypoints = max_pose_score_pair;           // max sample dimension for N timesteps

    // loop for every person in the image frame
    for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
    {
        // Supported for max 2 person. Limitation due to training samples
        if (person > 1)
            continue;

        // get person rectangle from keypoints
        const auto personRectangle = op::getKeypointsRectangle(keypoints, person, numberKeypoints, thresholdRectangle);
        // process if person is present
        if (personRectangle.area() > 0)
        {
            // Refer to openpose/include/openpose/pose/poseParameters.hpp for BODY parts index and PAIRS
            for (auto pair = 0 ; pair < pairs.size() ; pair+=2)
            {
                // obtain keypoint index per pair
                const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);

                // check if detected keypoints confidence is above threshold value
                if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                {
                    const int elementA = 0; // 0 is neck keypoint
                    const int elementB = pairs[pair+1];

                    // calculate distance metric between the pair points
                    if (_distance) // Eucledian distance (default)
                    {
                        pose_sample[(_frame_count_per_sample*pairs.size())+pair] = computeDistance(keypoints, person, elementA, elementB, true);
                    }
                    else // Cosine distance (default)
                    {
                        pose_sample[(_frame_count_per_sample*pairs.size())+pair] = computeDistance(keypoints, person, elementA, elementB, false);
                    }
                }
            }

            // keypoints: is a vector of pairs (x,y cordinates) and its detected confidence score
            // per sample consists of timesteps*keypoints. check if we got enough to make a sample.
            // eg. 5 timesteps each 40 dimension vector keypoints => gives 200 dimension sample
            if((_frame_count_per_sample==timesteps-1))
            {
                // generate samples
                if(_generate)
                {
                    // push a sample (timesteps*keypoints) to vector of samples
                    _pose_samples.push_back(pose_sample);
                    // push a label per sample
                    _pose_labels.push_back(label);
                    // count maintains number of pose_sample pushed
                    samples_count++;
                }
                else
                {
                    // check if we captured any sample
                    if(pose_sample.size()) {

                        // get pose sample prediction using keras classifier
                        auto pred = get_pose_predictions(_max_keypoints, pose_sample, _k_model);
                        int pred_label = std::get<0>(pred);
                        std::ostringstream score;
                        score << std::get<1>(pred);

                        // only use predicted classification with high confidence (> 90% is default)
                        if (std::get<1>(pred) > _classifier_threshold) {
                            _predicted_class_str[person] = class_name[pred_label];
                        } else {
                            _predicted_class_str[person] = class_name[0]; // unknown
                        }
                        _predicted_score_str[person] = score.str();

                        if (_debug) {
                            cout << "Predicted label: " << std::get<0>(pred) << " class:" << _predicted_class_str[person];
                            cout << " person:" << person << " score:" << std::get<1>(pred) << endl;
                        }
                    }
                }
                // clear sample vector at end of condition
                for (auto i = 0 ; i < timesteps*max_pose_score_pair ; i++)
                    pose_sample[i] = 0.0f;
            }

            // render to outputImage the class and score predicted
            if(!_generate)
            {
                string box_text = format("Pose = %s [score: %s]",
                                         _predicted_class_str[person].c_str(),
                                         _predicted_score_str[person].c_str());
                // And now put it into the image:
                putText(_outputImage, box_text, cv::Point(personRectangle.x, personRectangle.y),
                        FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255,255,0), 2.0);
            }
        }
    }

    // Debug information during sample generation
    if(_debug && _generate)
    {
        if(_pose_samples.size())
        {
            _label_count[label] = samples_count;
            if(!_frame_count_per_sample)
            {
                cout << endl << "Pose Summary:" << endl;
                cout << "pose_samples.size(): " << _pose_samples.size() << endl;
                cout << "pose_labels.size(): "<< _pose_labels.size() << endl;
                for (auto l=1; l < class_name.size(); l++)
                    if(class_name[l]!="")
                        cout << "Pose name: " << class_name[l] << " with label:" << l << " and count:" << _label_count[l] << endl;
            }
        }
    }
}

// Get samples from hand(L & R) keypoints
void pose_model::getSamplesFromHandKeypoints(cv::Mat &_outputImage, hand_sample_type &hand_sample, const Array<float>& keypoints,
                                             label_array &_label_count, const std::vector<unsigned int>& pairs, const int label,
                                             const float threshold, hand &class_name, int & _frame_count_per_sample, int & samples_count,
                                             hand & _predicted_class_str, hand & _predicted_score_str, const float _classifier_threshold,
                                             std::vector<double>& _hand_labels, std::vector<hand_sample_type>& _hand_samples,
                                             keypoints_types type, bool _generate, bool _distance, bool _debug, KerasModel &_k_model)
{
    const auto numberKeypoints = keypoints.getSize(1);  // keypoints vector size (dimension for one timestep)
    const auto thresholdRectangle = 0.1f;               // rectangle confidence threshold
    int _max_keypoints = max_hand_score_pair;           // max sample dimension for N timesteps

    // loop for every person in the image frame
    for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
    {
        // Supported for max 2 person. Limitation due to training samples
        if (person > 1)
            continue;

        // get person rectangle from keypoints
        const auto personRectangle = op::getKeypointsRectangle(keypoints, person, numberKeypoints, thresholdRectangle);
        // process if person is present
        if (personRectangle.area() > 0)
        {
            // Refer to openpose/include/openpose/hand/handParameters.hpp for BODY parts index and PAIRS
            for (auto pair = 0 ; pair < pairs.size() ; pair+=2)
            {
                // obtain keypoint index per pair
                const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);

                // check if detected keypoints confidence is above threshold value
                if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                {
                    const int elementA = 0; // 0 is wrist keypoint
                    const int elementB = pairs[pair + 1];

                    // calculate distance metric between the pair points
                    if (_distance) // Eucledian distance
                    {
                        hand_sample[(_frame_count_per_sample*pairs.size())+pair] = computeDistance(keypoints, person, elementA, elementB, true);
                    }
                    else // Cosine distance (default)
                    {
                        hand_sample[(_frame_count_per_sample*pairs.size())+pair] = computeDistance(keypoints, person, elementA, elementB, false);
                    }
                }
            }

            // keypoints: is a vector of pairs (x,y cordinates) and its detected confidence score
            // per sample consists of timesteps*keypoints. check if we got enough to make a sample.
            // eg. 5 timesteps each 40 dimension vector keypoints => gives 200 dimension sample
            if((_frame_count_per_sample==timesteps-1))
            {
                // generate samples
                if(_generate)
                {
                    // push a sample (timesteps*keypoints) to vector of samples
                    _hand_samples.push_back(hand_sample);
                    // push a label per sample
                    _hand_labels.push_back(label);
                    // count maintains number of hand_sample pushed
                    samples_count++;
                }
                else
                {
                    // check if we captured any sample
                    if(hand_sample.size())
                    {
                        // get hand sample prediction using keras classifier
                        auto pred = get_hand_predictions(_max_keypoints, hand_sample, _k_model);
                        int pred_label = std::get<0>(pred);
                        std::ostringstream score;
                        score << std::get<1>(pred);

                        // only use predicted classification with high confidence (> 90% is default)
                        if(std::get<1>(pred) > _classifier_threshold) {
                            _predicted_class_str[person] = class_name[pred_label];
                        } else {
                            _predicted_class_str[person] = class_name[0]; // unknown
                        }
                        _predicted_score_str[person] = score.str();

                        if(_debug) {
                            cout << "Predicted label: " << std::get<0>(pred) << " class:" << _predicted_class_str[person];
                            cout << " person:" << person << " score:" << std::get<1>(pred) << endl;
                        }
                    }
                }
                // clear sample vector at end of condition
                for (auto i = 0 ; i < timesteps*max_hand_score_pair ; i++)
                    hand_sample[i] = 0.0f;
            }

            // render to outputImage the class and score predicted
            if(!_generate)
            {
                string box_text;
                if(type == KP_LEFT_HAND)
                    box_text = format("Hand(L) = %s [score: %s]",
                                      _predicted_class_str[person].c_str(),
                                      _predicted_score_str[person].c_str());
                else
                    box_text = format("Hand(R) = %s [score: %s]",
                                      _predicted_class_str[person].c_str(),
                                      _predicted_score_str[person].c_str());
                // And now put it into the image:
                putText(_outputImage, box_text, cv::Point(personRectangle.x, personRectangle.y),
                        FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255,255,0), 2.0);
            }
        }
    }

    // Debug information during sample generation
    if(_debug && _generate)
    {
        if(_hand_samples.size())
        {
            _label_count[label] = samples_count;
            if(!_frame_count_per_sample)
            {
                if(type == KP_LEFT_HAND)
                    cout << endl << "Hand(L) Summary:" << endl;
                else
                    cout << endl << "Hand(R) Summary:" << endl;
                cout << "hand_samples.size(): " << _hand_samples.size() << endl;
                cout << "hand_labels.size(): "<< _hand_labels.size() << endl;
                for (auto l=1; l < class_name.size(); l++)
                    if(class_name[l]!="")
                        cout << "Hand name: " << class_name[l] << " type:" << type << " with label:" << l << " and count:" << _label_count[l] << endl;
            }
        }
    }
}

// Get samples from face keypoints
void pose_model::getSamplesFromFaceKeypoints(cv::Mat &_outputImage, face_sample_type &face_sample, const Array<float>& keypoints,
                                 label_array &_label_count, const std::vector<unsigned int>& pairs, const int _label,
                                 const float threshold, face &class_name, int & _frame_count_per_sample, int & samples_count,
                                 face & _predicted_class_str, face & _predicted_score_str, const float _classifier_threshold,
                                 std::vector<double>& _face_labels, std::vector<face_sample_type>& _face_samples,
                                 bool _generate, bool _distance, bool _debug, KerasModel &_k_model)
{
    const auto numberKeypoints = keypoints.getSize(1);  // keypoints vector size (dimension for one timestep)
    const auto thresholdRectangle = 0.1f;               // rectangle confidence threshold
    int _max_keypoints = max_face_score_pair;           // max sample dimension for N timesteps
    int last_unused_pair = 0;                           // unused pair index

    // loop for every person in the image frame
    for (auto person = 0 ; person < keypoints.getSize(0) ; person++)
    {
        // Supported for max 2 person. Limitation due to training samples
        if (person > 1)
            continue;

        // get person rectangle from keypoints
        const auto personRectangle = op::getKeypointsRectangle(keypoints, person, numberKeypoints, thresholdRectangle);
        // process if person is present
        if (personRectangle.area() > 0)
        {
            last_unused_pair = 0;

            // Refer to openpose/include/openpose/pose/poseParameters.hpp for BODY parts index and PAIRS
            // To reduce high-dimensionality ignore chin & jaw landmarks. Start from pair 16.
            for (auto pair = 16 ; pair < pairs.size() ; pair+=2)
            {
                // obtain keypoint index per pair
                const auto index1 = (person * numberKeypoints + pairs[pair]) * keypoints.getSize(2);
                const auto index2 = (person * numberKeypoints + pairs[pair+1]) * keypoints.getSize(2);

                // check if detected keypoints confidence is above threshold value
                if (keypoints[index1+2] > threshold && keypoints[index2+2] > threshold)
                {
                    const int elementA = 30; // 30 tip of nose
                    const int elementB = pairs[pair+1];

                    // calculate distance metric between the pair points
                    // for emotions face keypoints associated with face muscle movements (like mouth, eyebrows..)
                    // along with combination of l2 and cosine pair distance provided better classification
                    float l2 = computeDistance(keypoints, person, elementA, elementB, true);
                    double cosine = computeDistance(keypoints, person, elementA, elementB, false);

                    // store l2 and cosine distance at adjacent sample vector index
                    face_sample[(_frame_count_per_sample * _max_keypoints) + last_unused_pair] = l2;
                    face_sample[(_frame_count_per_sample * _max_keypoints) + last_unused_pair + 1] = cosine;
                    last_unused_pair += 2;
                }
            }

            // keypoints: is a vector of pairs (x,y cordinates) and its detected confidence score
            // per sample consists of timesteps*keypoints. check if we got enough to make a sample.
            // eg. 5 timesteps each 40 dimension vector keypoints => gives 200 dimension sample
            if((_frame_count_per_sample==timesteps-1))
            {
                // generate samples
                if(_generate)
                {
                    // push a sample (timesteps*keypoints) to vector of samples
                    _face_samples.push_back(face_sample);
                    // push a label per sample
                    _face_labels.push_back(_label);
                    // count maintains number of face_sample pushed
                    samples_count++;
                }
                else
                {
                    // check if we captured any sample
                    if(face_sample.size())
                    {
                        // get face sample prediction using keras classifier
                        auto pred = get_face_predictions(_max_keypoints, face_sample, _k_model);
                        int pred_label = std::get<0>(pred);
                        std::ostringstream score;
                        score << std::get<1>(pred);

                        // only use predicted classification with high confidence (> 90% is default)
                        if(std::get<1>(pred) > _classifier_threshold) {
                            _predicted_class_str[person] = class_name[pred_label];
                        } else {
                            _predicted_class_str[person] = class_name[0]; // normal
                        }
                        _predicted_score_str[person] = score.str();

                        if(_debug) {
                            cout << "Predicted label: " << std::get<0>(pred) << " class:" << _predicted_class_str[person];
                            cout << " person:" << person << " score:" << std::get<1>(pred) << endl;
                        }
                    }
                }
                // clear sample vector at end of condition
                for (auto i = 0 ; i < timesteps*max_face_score_pair ; i++)
                    face_sample[i] = 0.0f;
            }

            // render to outputImage the class and score predicted
            if(!_generate)
            {
                string box_text = format("Emotion = %s [score: %s]",
                                         _predicted_class_str[person].c_str(),
                                         _predicted_score_str[person].c_str());
                // And now put it into the image:
                putText(_outputImage, box_text, cv::Point(personRectangle.x, personRectangle.y),
                        FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255,255,0), 2.0);
            }

        }
    }

    // Debug information during sample generation
    if(_debug && _generate)
    {
        if(_face_samples.size())
        {
            _label_count[_label] = samples_count;
            if(!_frame_count_per_sample)
            {
                cout << endl << "Face Summary:" << endl;
                cout << "face_samples.size(): " << _face_samples.size() << endl;
                cout << "face_labels.size(): "<< _face_labels.size() << endl;
                for (auto l=1; l < class_name.size(); l++)
                    if(class_name[l]!="")
                        cout << "face name: " << class_name[l] << " with label:" << l << " and count:" << _label_count[l] << endl;
            }
        }
    }
}


