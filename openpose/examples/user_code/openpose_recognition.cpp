/*
 * File Description :   OpenPose based Pose, Gesture, Emotions Recognition and DLIB based Face Recognition
 * File Name        :   openpose_recognition.cpp
 * Author           :   Srini Ananthakrishnan
 * Date             :   09/23/2017
 * Github           :   https://github.com/srianant
 * Reference        :   OPENPOSE c++ examples (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
 *                  :   DLIB c++ examples (https://github.com/davisking/dlib)
 */

#include "openpose/user_code/dlib_model.hpp"
#include "openpose/user_code/pose_model.hpp"


// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             4,              "The logging level. Integer in the range [0, 255]. "
                                                        "0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library "
                                                        "messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(video_path,               "examples/media/video.avi",     "Process the desired image.");
DEFINE_int32(camera,                    0,              "The camera index for cv::VideoCapture. "
                                                        "Integer in the range [0, 9].");
DEFINE_string(camera_resolution,        "640x480",      "Size of the camera frames to ask for. eg: 1280x720, 640x480");
DEFINE_bool(debug,                      false,          "Enables keypoints detection debug");
DEFINE_int32(down_sample_frames,        0,              "Down sample. Camera frames processing interval. "
                                                        "Integer in the range [0, 50].");
DEFINE_bool(record,                     false,          "Record video");
DEFINE_bool(render,                     false,          "Render keypoints always. Rendering is GPU heavy, "
                                                        "might affect the performance");

// OpenPose Pose
DEFINE_bool(pose,                       false,          "Enables pose keypoint detection. "
                                                        "It will share some parameters from the body pose, "
                                                        " e.g. `model_folder`. Note that this will considerable "
                                                        "slow down the performance and increse"
                                                        " the required GPU memory. In addition, the greater number of "
                                                        "people on the image, the slower OpenPose will be.");
DEFINE_double(pose_classifier_threshold, 0.9,           "Pose classifier (svm) score of confidence");
DEFINE_double(pose_render_threshold,     0.05,          "Only estimated keypoints whose score confidences are "
                                                        "higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only "
                                                        "render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and "
                                                        "occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose,               1,              "Set to 0 for no rendering,1 for CPU rendering(slightly faster),"
                                                        "and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). "
                                                        "If rendering is enabled, it will"
                                                        " render both `outputData` and `cvOutputData` with the original "
                                                        "image and desired body part"
                                                        " to be shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models "
                                                        "(pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually "
                                                        "increases. If it is decreased,"
                                                        " the speed increases.");
DEFINE_string(resolution,               "640x480",     "The image resolution (display and output). Use \"-1x-1\" to "
                                                        "force the program to use the"
                                                        " default images resolution. eg: 1280x720 or 640x480");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(num_gpu,                   1,              "Total number of GPU device instance.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. "
                                                        "Initial scale is always 1. If you"
                                                        " want to change the initial scale, you actually want to "
                                                        "multiply the `net_resolution` by"
                                                        " your desired initial scale.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
// Pose Rendering
DEFINE_int32(part_to_show,              0,             "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the "
                                                        "original frame. If disabled, it"
                                                        " will only display the results.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. "
                                                        "1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame."
                                                        " 1 will only show the"
                                                        " heatmap, 0 will only show the frame. "
                                                        "Only valid for GPU rendering.");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters "
                                                        "from the body pose, e.g."
                                                        " `model_folder`. Note that this will considerable slow down "
                                                        "the performance and increse"
                                                        " the required GPU memory. In addition, the greater number of "
                                                        "people on the image, the"
                                                        " slower OpenPose will be.");
DEFINE_double(face_classifier_threshold, 0.9,            "Face classifier (svm) score of confidence");
DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the "
                                                        "face keypoints.");
DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16. Analogous to `net_resolution` but applied to "
                                                        "the face keypoint detector."
                                                        " 320x320 usually works fine while giving a substantial speed "
                                                        "up when multiple faces on the"
                                                        " image.");
// Face Rendering
DEFINE_int32(render_face,               -1,             "Analogous to `render_pose` but applied to the face. "
                                                        "Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_bool(with_video,                 false,          "Use /openpose/test_video.mp4 video");
DEFINE_bool(dlib_face,                  false,          "Use DLIB for face recognition");

// OpenPose Hand
DEFINE_double(hand_classifier_threshold, 0.9,           "Hand classifier (svm) score of confidence");
DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` "
                                                        "but applied to the hand keypoint"
                                                        " detector.");
// Hand Rendering
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters"
                                                        " from the body pose, e.g."
                                                        " `model_folder`. Analogously to `--face`, it will also "
                                                        "slow down the performance, increase"
                                                        " the required GPU memory and its speed depends on "
                                                        "the number of people.");
DEFINE_double(hand_render_threshold,    0.2,            "Analogous to `render_threshold`, but applied "
                                                        "to the hand keypoints.");
DEFINE_int32(render_hand,               -1,             "Analogous to `render_pose` but applied to the hand. "
                                                        "Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(alpha_hand,               0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(alpha_heatmap_hand,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");


// Common routines for parameter parsing
op::RenderMode gflagToRenderMode(const int renderFlag, const int renderPoseFlag = -2)
{
    if (renderFlag == -1 && renderPoseFlag != -2)
        return gflagToRenderMode(renderPoseFlag, -2);
    else if (renderFlag == 0)
        return op::RenderMode::None;
    else if (renderFlag == 1)
        return op::RenderMode::Cpu;
    else if (renderFlag == 2)
        return op::RenderMode::Gpu;
    else
    {
        op::error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
        return op::RenderMode::None;
    }
}

op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (poseModeString == "COCO")
        return op::PoseModel::COCO_18;
    else if (poseModeString == "MPI")
        return op::PoseModel::MPI_15;
    else if (poseModeString == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        return op::PoseModel::COCO_18;
    }
}

// Google flags into program variables
std::tuple<op::Point<int>, op::Point<int>, op::Point<int>, op::PoseModel> gflagsToOpParameters()
{
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    op::Point<int> outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
    cout << "resolution: " << outputSize.x << "x" << outputSize.y << endl;
    op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ",
               __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    op::Point<int> netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.x, &netInputSize.y);
    cout << "net_resolution: " << netInputSize.x << "x" << netInputSize.y << endl;
    op::checkE(nRead, 2,
               "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, "
               "should be e.g., 656x368 (multiples of 16)",
               __LINE__, __FUNCTION__, __FILE__);
    // netOutputSize
    const auto netOutputSize = netInputSize;
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
        op::error("Uncompatible flag configuration: scale_gap must be greater than "
                          "0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
    // Logging and return result
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
}


// Main routine for openpose and dlib based recognition
int openPoseRecognition()
{
    op::log("OpenPose/DLIB Gesture, Action and Face Recognition.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // -------------------------------------------------------
    // Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing
    // -------------------------------------------------------
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255,
              "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

    if((FLAGS_face || FLAGS_dlib_face) && FLAGS_hand)
        if((int)FLAGS_num_gpu < 2)
            op::error("WARNING: Due to memory constraint FACE and HAND detections not allowed simultaneously..!!");

    if(!(FLAGS_pose || FLAGS_face || FLAGS_hand || FLAGS_dlib_face)) {
        cout << "=====================================================================================================" << endl;
        cout << "WARNING: Please provide at least one of the supported detections pose/face/hand/dlib_face. eg: --pose" << endl;
        cout << "=====================================================================================================" << endl;
    }

    // -------------------------------------------------------
    // Read Google flags (user defined configuration)
    // -------------------------------------------------------
    op::Point<int> outputSize;
    op::Point<int> netInputSize;
    op::Point<int> netOutputSize;
    op::PoseModel poseModel;
    std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters();

    // -------------------------------------------------------
    // Initialize all required classes
    // -------------------------------------------------------
    op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap};
    op::CvMatToOpOutput cvMatToOpOutput{outputSize};
    op::OpOutputToCvMat opOutputToCvMat{outputSize};

    // Instantiate DLIB and POSE model classes
    pose_model pm;
    dlib_model dm;

    // Pose Extractor
    std::shared_ptr<op::PoseExtractor> poseExtractorPtr =
            std::make_shared<op::PoseExtractorCaffe>(netInputSize, netOutputSize, outputSize, FLAGS_num_scales,
                                                     poseModel, FLAGS_model_folder, FLAGS_num_gpu_start);
    // Pose Renderer
    op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, poseExtractorPtr,
                                  (float)FLAGS_pose_render_threshold, !FLAGS_disable_blending,
                                  (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
                                  (unsigned int)FLAGS_part_to_show, gflagToRenderMode(FLAGS_render_pose)};

    // Hand detector
    op::HandDetector handDetector(poseModel);

    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
    cout << "handNetInputSize: " << handNetInputSize.x << "x" << handNetInputSize.y << endl;

    // Hand Extractor
    std::shared_ptr<op::HandExtractor> handExtractor =
            std::make_shared<op::HandExtractor>(handNetInputSize, handNetInputSize, FLAGS_model_folder,
                                                FLAGS_num_gpu_start);
    // Hand Renderer
    op::HandRenderer handRenderer{handNetInputSize, (float)FLAGS_hand_render_threshold, (float)FLAGS_alpha_hand,
                                  (float)FLAGS_alpha_heatmap_hand, gflagToRenderMode(FLAGS_render_pose)};

    // Face detector
    op::FaceDetector faceDetector(poseModel);

    // Face Extractor
    op::Point<int> faceNetInputSize;
    auto nRead = sscanf(FLAGS_face_net_resolution.c_str(), "%dx%d", &faceNetInputSize.x, &faceNetInputSize.y);
    op::checkE(nRead, 2, "Error, face net resolution format (" +  FLAGS_face_net_resolution
               + ") invalid, should be e.g., 368x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    cout << "face_net_resolution: " << faceNetInputSize.x << "x" << faceNetInputSize.y << endl;

    // Input vs Output size must be same
    op::FaceExtractor faceExtractor(faceNetInputSize, faceNetInputSize, FLAGS_model_folder, FLAGS_num_gpu_start);

    // Face Renderer
    op::FaceRenderer faceRenderer{faceNetInputSize,(float)FLAGS_face_render_threshold, (float)FLAGS_alpha_pose,
                                  (float)FLAGS_alpha_heatmap, gflagToRenderMode(FLAGS_render_face, FLAGS_render_pose)};

    // -------------------------------------------------------
    // Initialize resources on desired threads
    // -------------------------------------------------------
    if(FLAGS_pose || FLAGS_face || FLAGS_hand || FLAGS_dlib_face)
    {
        poseExtractorPtr->initializationOnThread();
        poseRenderer.initializationOnThread();
    }

    if(FLAGS_hand)
    {
        handExtractor->initializationOnThread();
        handRenderer.initializationOnThread();
    }

    if(FLAGS_face || FLAGS_dlib_face)
    {
        faceExtractor.initializationOnThread();
        faceRenderer.initializationOnThread();
    }

    // -------------------------------------------------------
    // Read/Capture image, error if empty
    // -------------------------------------------------------
    // openCV image matrix
    cv::Mat inputImage;
    cv::Mat outputImage;
    cv::Mat origImage;

    // video/image display windows
    cv::namedWindow("pose image");
    cv::moveWindow("pose image", 640,20);
    if(FLAGS_dlib_face) {
        cv::namedWindow("new face");
        cv::moveWindow("new face", 20,20);
    }

    // video capture object to acquire
    cv::VideoCapture capture;
    if(FLAGS_with_video) // read video from file
    {
        // capture from video file
        capture.open(FLAGS_video_path);
        if(!capture.isOpened())
        {
            cout << "Error can't open the file"<<endl;
            return 0;
        }
    }
    else // capture image from camera
    {
        // open capture object at location zero (default location for webcam)
        capture.open(FLAGS_camera); // default FLAGS_camera index is 0

        // camera_resolution
        op::Point<int> cameraResolution;
        nRead = sscanf(FLAGS_camera_resolution.c_str(), "%dx%d", &cameraResolution.x, &cameraResolution.y);
        op::checkE(nRead, 2, "Error, camera resolution format "
                           "(" +  FLAGS_camera_resolution + ") invalid, should be e.g., width x height - 1280x720",
                   __LINE__, __FUNCTION__, __FILE__);
        cout << "Camera Resolution set to: " << cameraResolution.x << "x" << cameraResolution.y << endl;
        //set height and width of capture frame
        capture.set(CV_CAP_PROP_FRAME_WIDTH,cameraResolution.x);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT,cameraResolution.y);
    }

    // Video writer object
    VideoWriter outputVideo;
    // Write to video file in openpose root directory
    if (FLAGS_record) {
        const string NAME = "gesture_video.avi";   // Form the new name with container
        //int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

        Size S = Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                      (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));

        outputVideo.open(NAME, CV_FOURCC('D', 'I', 'V', 'X'), 5.5, S, true);

        if (!outputVideo.isOpened()) {
            cout << "Could not open the output video for write: " << endl;
            return -1;
        }
    }

    // -------------------------------------------------------
    // Variables declaration
    // -------------------------------------------------------
    int frame_down_sample_count = 0; // frames down sample count
    int frame_count_per_sample = 0;  // frame counts per sample
    bool generate_pose = false;
    bool generate_face = false;
    bool generate_hand = false;
    bool pause = false;

    // pose related
    pose pose_names;                            // vector of pose names (eg. sitting, standing..)
    pose_sample_type m_pose_sample;             // vector of pose sample
    std::vector<pose_sample_type> pose_samples; // vector of m_pose_sample vectors
    std::vector<double> pose_labels;            // vector of pose labels (eg. 1,2,3...)
    pose pose_predicted_class_str;              // vector of per person predicted class or pose names
    pose pose_predicted_score_str;              // vector of per person predicted score.
                                                // predicted score close to 1 indicates higher classification confidence
    label_array pose_label_count;               // vector of generated pose samples label count
    int pose_label = 0;                         // assigned pose label for new sample generation
    int pose_count = 0;                         // count of samples for assigned pose_label

    // face emotions related
    face face_names;                            // vector of face emotions names (eg. normal, sad, happy...)
    face_sample_type m_face_sample;             // vector of face sample
    std::vector<face_sample_type> face_samples; // vector of m_face_sample vectors
    std::vector<double> face_labels;            // vector of face emotions labels (eg. 1,2,3...)
    face face_predicted_class_str;              // vector of per person predicted class or face names
    face face_predicted_score_str;              // vector of per person predicted score.
                                                // predicted score close to 1 indicates higher classification confidence
    label_array face_label_count;               // vector of generated face samples label count
    int face_label = 0;                         // assigned face label for new sample generation
    int face_count = 0;                         // count of samples for assigned face_label
    bool face_capture = false;                  // DLIB flag to capture new face

    // hand related
    hand hand_names;                                    // vector of hand gesture names (eg. fist, victory, stop...)
    hand_sample_type m_left_hand_sample;                // vector of left hand sample
    hand_sample_type m_right_hand_sample;               // vector of right hand sample
    std::vector<hand_sample_type> left_hand_samples;    // vector of m_left_hand_sample vectors
    std::vector<hand_sample_type> right_hand_samples;   // vector of m_right_hand_sample vectors
    std::vector<double> left_hand_labels;               // vector of left hand labels (eg. 1,2,3...)
    std::vector<double> right_hand_labels;              // vector of right hand labels (eg. 1,2,3...)
    hand left_hand_predicted_class_str;                 // vector of per person predicted class or hand gesture names(L)
    hand right_hand_predicted_class_str;                // vector of per person predicted class or hand gesture names(R)
    hand left_hand_predicted_score_str;                 // vector of per person predicted score.(L)
    hand right_hand_predicted_score_str;                // vector of per person predicted score.(R)
    label_array hand_label_count;                       // vector of generated hand samples label count
    int hand_label = 0;                                 // assigned hand label for new sample generation
    int hand_count = 0;                                 // count of samples for assigned hand_label

    // -------------------------------------------------------
    // Initialize all vectors with default
    // -------------------------------------------------------
    for (auto i = 0 ; i < max_pose_count ; i++)
    {
        if(i == 0){
            // reserve index 0 for unknown
            pose_names[i] = "unknown";
            face_names[i] = "unknown";
            hand_names[i] = "unknown";
        } else {
            pose_names[i] = "";
            face_names[i] = "";
            hand_names[i] = "";
        }

        pose_predicted_class_str[i] = "";
        face_predicted_class_str[i] = "";
        left_hand_predicted_class_str[i] = "";
        right_hand_predicted_class_str[i] = "";

        pose_predicted_score_str[i] = "";
        face_predicted_score_str[i] = "";
        left_hand_predicted_score_str[i] = "";
        right_hand_predicted_score_str[i] = "";

        pose_label_count[i] = 0;
        face_label_count[i] = 0;
        hand_label_count[i] = 0;
    }

    // initialize sample vector
    for (auto i=0; i < timesteps*max_pose_score_pair; i++)
        m_pose_sample[i] = 0.0f;
    for (auto i=0; i < timesteps*max_face_score_pair; i++)
        m_face_sample[i] = 0.0f;
    for (auto i=0; i < timesteps*max_hand_score_pair; i++)
    {
        m_left_hand_sample[i] = 0.0f;
        m_right_hand_sample[i] = 0.0f;
    }

    // -------------------------------------------------------
    // Display user menu depending on user flags
    // -------------------------------------------------------
    if(FLAGS_pose || FLAGS_hand || FLAGS_face || FLAGS_dlib_face) {
        cout << "Push following keys:" << endl;
        cout << "p for pause sample generation" << endl;
    }

    if(FLAGS_pose)
        cout << "g for generating pose samples" << endl;

    if(FLAGS_hand)
        cout << "a for generating hands samples" << endl;

    if(FLAGS_face)
        cout << "f for generating face samples" << endl;

    if(FLAGS_dlib_face)
        cout << "k for start capturing new face" << endl;

    if(FLAGS_pose || FLAGS_hand || FLAGS_face)
        cout << "t for train samples" << endl;

    if(FLAGS_pose || FLAGS_hand || FLAGS_face || FLAGS_dlib_face) {
        cout << "c for continue camera feed" << endl;
        cout << "h for display key commands" << endl;
    }
    cout << "q for quit program" << endl;

    // -------------------------------------------------------
    // Create train_data directory if not present
    // -------------------------------------------------------
    const int dir_err = system("mkdir -p train_data");
    if (-1 == dir_err)
    {
        cout << "Error creating directory..!!" << endl;
        return (0);
    }

    // -------------------------------------------------------
    // Load names, labels, samples vectors
    // -------------------------------------------------------
    // Load Pose Vectors
    if(FLAGS_pose)
    {
        if (std::ifstream("train_data/pose/pose_names.txt"))
        {
            std::ifstream ifile{"train_data/pose/pose_names.txt"};
            text_iarchive ia{ifile}; ia >> pose_names;

            if(FLAGS_debug)
                cout << "pose names size...:" << pose_names.size() << endl;
            for (auto i=0; i < pose_names.size(); i++)
            {
                if (pose_names[i]!="")
                {
                    if(FLAGS_debug)
                        cout << "pose Label:" << i << " name:" << pose_names[i].c_str() << endl;
                    pose_label = i;
                }
            }
        }
        else
            cout << "WARNING: Saved pose table not found. Generate pose samples" << endl;

        if (std::ifstream("train_data/pose/pose_samples.txt")) {
            std::ifstream ifile{"train_data/pose/pose_samples.txt"};
            text_iarchive ia{ifile}; ia >> pose_samples;
        }else {
            cout << "WARNING: Saved pose samples not found. Generate pose samples" << endl;
        }

        if (std::ifstream("train_data/pose/pose_labels.txt")) {
            std::ifstream ifile{"train_data/pose/pose_labels.txt"};
            text_iarchive ia{ifile}; ia >> pose_labels;
        }else {
            cout << "WARNING: Saved pose labels not found. Generate pose labels" << endl;
        }
    }

    // Load Face Vectors
    if(FLAGS_face)
    {
        if(std::ifstream("train_data/face/face_names.txt"))
        {
            std::ifstream ifile{"train_data/face/face_names.txt"};
            text_iarchive ia{ifile}; ia >> face_names;

            if(FLAGS_debug)
                cout << "face names size...:" << face_names.size() << endl;
            for (auto i=0; i < face_names.size(); i++)
            {
                if (face_names[i]!="")
                {
                    if(FLAGS_debug)
                        cout << "face Label:" << i << " name:" << face_names[i].c_str() << endl;
                    face_label = i;
                }
            }
        }
        else {
            cout << "WARNING: Saved face table not found. Generate face samples" << endl;
        }

        if (std::ifstream("train_data/face/face_samples.txt")) {
            std::ifstream ifile{"train_data/face/face_samples.txt"};
            text_iarchive ia{ifile}; ia >> face_samples;
        }else {
            cout << "WARNING: Saved face samples not found. Generate face samples" << endl;
        }

        if (std::ifstream("train_data/face/face_labels.txt")) {
            std::ifstream ifile{"train_data/face/face_labels.txt"};
            text_iarchive ia{ifile}; ia >> face_labels;
        }else {
            cout << "WARNING: Saved face labels not found. Generate face labels" << endl;
        }
    }

    // Load Hand Vectors
    if(FLAGS_hand)
    {
        if (std::ifstream("train_data/hand/hand_names.txt")) {
            std::ifstream ifile{"train_data/hand/hand_names.txt"};
            text_iarchive ia{ifile}; ia >> hand_names;

            if(FLAGS_debug)
                cout << "Hand names size...:" << hand_names.size() << endl;
            for (auto i=0; i < hand_names.size(); i++)
            {
                if (hand_names[i]!="")
                {
                    if(FLAGS_debug)
                        cout << "Hands Label:" << i << " name:" << hand_names[i].c_str() << endl;
                    hand_label = i;
                }
            }
        }
        else {
            cout << "WARNING: Saved hand names not found. Generate hand names" << endl;
        }

        if (std::ifstream("train_data/hand/left_hand_samples.txt")) {
            std::ifstream ifile{"train_data/hand/left_hand_samples.txt"};
            text_iarchive ia{ifile}; ia >> left_hand_samples;
        }else {
            cout << "WARNING: Saved left hand samples not found. Generate left hand samples" << endl;
        }

        if (std::ifstream("train_data/hand/left_hand_labels.txt")) {
            std::ifstream ifile{"train_data/hand/left_hand_labels.txt"};
            text_iarchive ia{ifile}; ia >> left_hand_labels;
        }else {
            cout << "WARNING: Saved left hand labels not found. Generate left hand labels" << endl;
        }

        if (std::ifstream("train_data/hand/right_hand_samples.txt")) {
            std::ifstream ifile{"train_data/hand/right_hand_samples.txt"};
            text_iarchive ia{ifile}; ia >> right_hand_samples;
        }else {
            cout << "WARNING: Saved right hand samples not found. Generate right hand samples" << endl;
        }

        if (std::ifstream("train_data/hand/right_hand_labels.txt")) {
            std::ifstream ifile{"train_data/hand/right_hand_labels.txt"};
            text_iarchive ia{ifile}; ia >> right_hand_labels;
        }else {
            cout << "WARNING: Saved right hand labels not found. Generate right hand labels" << endl;
        }
    }

    // -------------------------------------------------------
    // Initialize Keras RNN LSTM Classifier model
    // -------------------------------------------------------
    KerasModel k_pose_model;
    KerasModel k_hand_model;
    KerasModel k_face_model;
    if(FLAGS_pose) {
        if (std::ifstream("train_data/pose/pose.model")) {
            cout << "Loading keras pose model...!!" << endl;
            k_pose_model.LoadModel("train_data/pose/pose.model");
            cout << "Done loading pose keras lstm classifier model" << endl;
        } else
            cout << "WARNING: No pose classifier model found..!!" << endl;
    }
    if(FLAGS_hand) {
        if (std::ifstream("train_data/hand/hand.model")) {
            cout << "Loading keras hand model...!!" << endl;
            k_hand_model.LoadModel("train_data/hand/hand.model");
            cout << "Done loading hand keras lstm classifier model" << endl;
        } else
            cout << "WARNING: No hand classifier model found..!!" << endl;
    }
    if(FLAGS_face) {
        if (std::ifstream("train_data/face/face.model")) {
            cout << "Loading keras face model...!!" << endl;
            k_face_model.LoadModel("train_data/face/face.model");
            cout << "Done loading face keras lstm classifier model" << endl;
        } else
            cout << "WARNING: No face classifier model found..!!" << endl;
    }

    // -------------------------------------------------------
    // DLIB train face images in folder train_data/dlib_faces
    // -------------------------------------------------------
    if(FLAGS_dlib_face)
    {
        // Train face images saved in folder train_data/dlib_faces
        if(!dm.facerec_dlib_train_face_images(0))
        {
            cout << "ERROR: Training sample faces" << endl;
            return (0);
        }
    }

    // -----------------------------------------------------------
    // LOOP continuously for processing image and user pushed keys
    // -----------------------------------------------------------
    while(1)
    {
        // -------------------------------------------------------
        // Check if user pushed any keys and process the request
        // -------------------------------------------------------
        auto k = cv::waitKey(50);

        if (k==113) // key "q" to quit program
            break;

        switch(k)
        {
            case 97: // key "a" for generating actions samples
            {
                if(FLAGS_hand)
                {
                    cout << "Please enter a hand actions name (like holding cup, phone, pen):" << endl;
                    std::string name = dm.readStdIn(20); // check user input for 10 sec
                    if(!name.empty())
                    {
                        hand_count = 0;
                        hand_label++;
                        hand_names[hand_label] = name;
                        generate_hand = true;
                        cout << "Generating actions " << name << " samples" << endl;
                        pause = false;
                    }
                    else
                    {
                        cout << "No key input for 10 sec...try again by pressing a" << endl;
                        generate_hand = false;
                    }
                    generate_pose = false;
                    generate_face = false;
                }
            }
            break;

            case 103: // key "g" for generating pose samples
            {
                if(FLAGS_pose)
                {
                    cout << "Please enter a pose name (like sitting, standing, etc):" << endl;
                    std::string name = dm.readStdIn(20); // check user input for 10 sec
                    if(!name.empty())
                    {
                        pose_count = 0;
                        pose_label++;
                        pose_names[pose_label] = name;
                        generate_pose = true;
                        cout << "Generating pose " << name << " samples" << endl;
                        pause = false;
                    }
                    else
                    {
                        cout << "No key input for 10 sec...try again by pressing g" << endl;
                        generate_pose = false;
                    }
                    generate_hand = false;
                    generate_face = false;
                }
            }
            break;

            case 102: // key "f" for generating face samples
            {
                if(FLAGS_face || FLAGS_dlib_face)
                {
                    if(FLAGS_dlib_face)
                    {
                        generate_pose = false;
                        generate_face = false;
                        generate_hand = false;
                        pause = false;
                    }
                    else // openpose face emotions
                    {
                        cout << "Please enter a face emotion name:" << endl;
                        std::string name = dm.readStdIn(20); // check user input for 10 sec
                        if(!name.empty())
                        {
                            face_count = 0;
                            face_label++;
                            face_names[face_label] = name;
                            generate_face = true;
                            cout << "Generating face emotion " << name << " samples";
                            cout << " with label:" << face_label << endl;
                            pause = false;
                        }
                        else
                        {
                            cout << "No key input for 10 sec...try again by pressing f" << endl;
                            generate_face = false;
                        }
                        generate_pose = false;
                        generate_hand = false;
                    }
                }
            }
            break;

            case 104: // key "h" print commands
                if(FLAGS_pose || FLAGS_hand || FLAGS_face || FLAGS_dlib_face) {

                    cout << "Push following keys:" << endl;
                    cout << "p for pause sample generation" << endl;

                    if (FLAGS_pose)
                        cout << "g for generating pose samples" << endl;
                    if (FLAGS_hand)
                        cout << "a for generating hands samples" << endl;
                    if (FLAGS_face)
                        cout << "f for generating face samples" << endl;
                    if (FLAGS_dlib_face)
                        cout << "k for start capturing new face" << endl;
                    if (FLAGS_pose || FLAGS_hand || FLAGS_face)
                        cout << "t for train samples" << endl;
                    cout << "c for continue camera feed" << endl;
                    cout << "h for display key commands" << endl;
                    cout << "q for quit program" << endl;
                }
            break;

            case 107: // key "k" for start capturing new face
                if(FLAGS_dlib_face)
                {
                    if(!face_capture) {
                        face_capture = true;
                        cout << "Capturing new face...: ON" << endl;
                    } else {
                        face_capture = false;
                        cout << "Capturing new face...: OFF" << endl;
                    }
                    pause = false;
                };
            break;

            case 112: // key "p" for pause source
                pause = !pause;
                if(pause)
                    cout << "Pausing generation and recognition...!!. Press p to continue" << endl;
            break;

            case 116: // key "t" prepare train samples
            {
                if(FLAGS_pose)
                {
                    if(pose_samples.size())
                    {
                        cout << "Prepare pose train samples..." << endl;

                        std::ofstream out_sample("train_data/pose/pose_samples_raw.txt");
                        std::ofstream out_label("train_data/pose/pose_labels_raw.txt");

                        cout << "Pose Summary:" << endl;
                        cout << "samples.size(): "<< pose_samples.size() << endl;
                        cout << "labels.size(): "<< pose_labels.size() << endl;

                        // Save raw samples & labels. This will be used to train keras python LSTM classifier model
                        for(auto i = 0; i < pose_samples.size(); i++)
                        {
                            for (auto j = 0; j < pose_samples[i].size(); j++)
                                out_sample << pose_samples[i][j] << " ";
                            out_sample << endl;
                            out_label << pose_labels[i] << endl;
                        }

                        std::ofstream pose_names_ofile{"train_data/pose/pose_names.txt"};
                        text_oarchive pose_names_oa{pose_names_ofile}; pose_names_oa << pose_names;

                        std::ofstream pose_samples_ofile{"train_data/pose/pose_samples.txt"};
                        text_oarchive pose_samples_oa{pose_samples_ofile}; pose_samples_oa << pose_samples;

                        std::ofstream pose_labels_ofile{"train_data/pose/pose_labels.txt"};
                        text_oarchive pose_labels_oa{pose_labels_ofile}; pose_labels_oa << pose_labels;
                    }
                    else
                    {
                        cout << "No pose samples..!!" << endl;
                    }
                }

                if(FLAGS_face)
                {
                    if(face_samples.size())
                    {
                        cout << "Prepare face train samples..." << endl;

                        std::ofstream out_sample("train_data/face/face_samples_raw.txt");
                        std::ofstream out_label("train_data/face/face_labels_raw.txt");

                        cout << "Face Summary:" << endl;
                        cout << "samples.size(): "<< face_samples.size() << endl;
                        cout << "labels.size(): "<< face_labels.size() << endl;

                        // Save raw samples & labels. This will be used to train keras python LSTM classifier model
                        for(auto i = 0; i < face_samples.size(); i++)
                        {
                            for (auto j = 0; j < face_samples[i].size(); j++)
                                out_sample << face_samples[i][j] << " ";
                            out_sample << endl;
                            out_label << face_labels[i] << endl;
                        }

                        std::ofstream face_names_ofile{"train_data/face/face_names.txt"};
                        text_oarchive face_names_oa{face_names_ofile}; face_names_oa << face_names;

                        std::ofstream face_samples_ofile{"train_data/face/face_samples.txt"};
                        text_oarchive face_samples_oa{face_samples_ofile}; face_samples_oa << face_samples;

                        std::ofstream face_labels_ofile{"train_data/face/face_labels.txt"};
                        text_oarchive face_labels_oa{face_labels_ofile}; face_labels_oa << face_labels;
                    }
                    else
                    {
                        cout << "No face samples..!!" << endl;
                    }
                }

                if(FLAGS_hand)
                {
                    if (left_hand_samples.size() || right_hand_samples.size())
                    {
                        cout << "Prepare hand train samples..." << endl;

                        std::ofstream ofile{"train_data/hand/hand_names.txt"};
                        text_oarchive oa{ofile}; oa << hand_names;
                        cout << "Hand Summary:" << endl;
                    }
                    if(left_hand_samples.size())
                    {
                        std::ofstream out_sample("train_data/hand/hand_samples_raw.txt");
                        std::ofstream out_label("train_data/hand/hand_labels_raw.txt");

                        cout << "left_hand_samples.size(): "<< left_hand_samples.size() << endl;
                        cout << "left_hand_labels.size(): "<< left_hand_labels.size() << endl;

                        // Save raw samples & labels. This will be used to train keras python LSTM classifier model
                        for(auto i = 0; i < left_hand_samples.size(); i++)
                        {
                            for (auto j = 0; j < left_hand_samples[i].size(); j++)
                                out_sample << left_hand_samples[i][j] << " ";
                            out_sample << endl;
                            out_label << left_hand_labels[i] << endl;
                        }

                        // Serialize samples and labels
                        std::ofstream lhs_ofile{"train_data/hand/left_hand_samples.txt"};
                        text_oarchive lhs_oa{lhs_ofile}; lhs_oa << left_hand_samples;

                        std::ofstream lhl_ofile{"train_data/hand/left_hand_labels.txt"};
                        text_oarchive lhl_oa{lhl_ofile}; lhl_oa << left_hand_labels;
                    }
                    else
                    {
                        cout << "No left hand samples..!!" << endl;
                    }

                    if(right_hand_samples.size())
                    {
                        std::ofstream out_sample("train_data/hand/hand_samples_raw.txt");
                        std::ofstream out_label("train_data/hand/hand_labels_raw.txt");

                        cout << "right_hand_samples.size(): "<< right_hand_samples.size() << endl;
                        cout << "right_hand_labels.size(): "<< right_hand_labels.size() << endl;

                        // Save raw samples & labels. This will be used to train keras python LSTM classifier model
                        for(auto i = 0; i < right_hand_samples.size(); i++)
                        {
                            for (auto j = 0; j < right_hand_samples[i].size(); j++)
                                out_sample << right_hand_samples[i][j] << " ";
                            out_sample << endl;
                            out_label << right_hand_labels[i] << endl;
                        }

                        std::ofstream rhs_ofile{"train_data/hand/right_hand_samples.txt"};
                        text_oarchive rhs_oa{rhs_ofile}; rhs_oa << right_hand_samples;

                        std::ofstream rhl_ofile{"train_data/hand/right_hand_labels.txt"};
                        text_oarchive rhl_oa{rhl_ofile}; rhl_oa << right_hand_labels;
                    }
                    else
                    {
                        cout << "No right hand samples..!!" << endl;
                    }
                }

                if(FLAGS_pose || FLAGS_hand || FLAGS_face || FLAGS_dlib_face) {
                    cout << "Done copying training samples to train_data directory...!!" << endl;
                    cout << "Run examples/user_code/python/rnn_lstm_classifier.ipynb";
                    cout << " python notebook to generate keras models..!!" << endl;
                    generate_pose = false;
                    generate_face = false;
                    generate_hand = false;
                    face_capture = false;
                    pause = false;
                }
            }
            break;

            case 99: // key "c" for continue camera feed
            {
                generate_pose = false;
                generate_face = false;
                generate_hand = false;
                face_capture = false;
                pause = false;
            }
            break;

            default:
            break;
        }
        k = 0;

        // -------------------------------------------------------
        // Read captured image to matrix
        // -------------------------------------------------------
        capture.read(inputImage);

        if(inputImage.empty())
            op::error("Could not open or find the image: " + FLAGS_video_path, __LINE__, __FUNCTION__, __FILE__);


        // -------------------------------------------------------
        // Check and handle for down sampling of frames
        // -------------------------------------------------------
        if(frame_down_sample_count < FLAGS_down_sample_frames)
        {
            frame_down_sample_count++;
            continue;
        }
        else
        {
           frame_down_sample_count = 0;
        }

        // -------------------------------------------------------
        // Format input image to OpenPose input and output formats
        // -------------------------------------------------------
        op::Array<float> netInputArray;
        std::vector<float> scaleRatios;
        std::tie(netInputArray, scaleRatios) = cvMatToOpInput.format(inputImage);
        double scaleInputToOutput;
        op::Array<float> outputArray;
        std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);

        // -------------------------------------------------------
        // POSE, HAND, FACE ESTIMATION AND RENDERING
        // -------------------------------------------------------

        // Handle user pause and copy inputImage to outputImage
        if (pause)
        {
            // OpenPose output format to cv::Mat
            outputImage = opOutputToCvMat.formatToCvMat(outputArray);
            origImage = outputImage;
        }
        else
        {
            Array<float> poseKeypoints;
            float scaleNetToOutput = 0.0f;

            // Estimate poseKeypoints. Pose keypoints are required to compute hand and face keypoints
            if(FLAGS_pose || FLAGS_hand || FLAGS_face || FLAGS_dlib_face) {
                poseExtractorPtr->forwardPass(netInputArray, {inputImage.cols, inputImage.rows}, scaleRatios);
                poseKeypoints = poseExtractorPtr->getPoseKeypoints();
                scaleNetToOutput = poseExtractorPtr->getScaleNetToOutput();
            }

            std::vector < std::array < Rectangle < float > , 2 >> handsRectangles;
            std::array<Array<float>, 2> handKeypoints;

            // Estimate handKeypoints
            if (FLAGS_hand) {
                handsRectangles = handDetector.trackHands(poseKeypoints, scaleInputToOutput);
                handExtractor->forwardPass(handsRectangles, inputImage, scaleInputToOutput);
                handKeypoints = handExtractor->getHandKeypoints();
            }

            // Render pose keypoints
            if ((generate_pose && FLAGS_pose) || (FLAGS_pose && FLAGS_render))
                poseRenderer.renderPose(outputArray, poseKeypoints, scaleNetToOutput);

            // Render hand keypoints
            if ((generate_hand && FLAGS_hand) || (FLAGS_hand && FLAGS_render))
                handRenderer.renderHand(outputArray, handKeypoints);


            std::vector <Rectangle<float>> faceRectangles;
            Array<float> faceKeypoints;

            // Estimate faceKeypoints
            if (FLAGS_face || FLAGS_dlib_face) {
                // faceDetector return Rectangle for all faces
                faceRectangles = faceDetector.detectFaces(poseKeypoints, scaleInputToOutput);
                faceExtractor.forwardPass(faceRectangles, inputImage, scaleInputToOutput);
                faceKeypoints = faceExtractor.getFaceKeypoints();
                // Render face keypoints
                if ((generate_face && !FLAGS_dlib_face) || (FLAGS_face && FLAGS_render))
                    faceRenderer.renderFace(outputArray, faceKeypoints);
            }

            // Run DLIB face recognition for face rectangles detected using openpose faceDetector
            if(FLAGS_dlib_face) {
                outputImage = inputImage;
                dm.dlib_facerec_from_faceRectangles(outputImage, faceKeypoints, frame_count_per_sample,
                                                    &face_capture, FLAGS_debug);
            }
            else {
                if(FLAGS_pose || FLAGS_hand || FLAGS_face) {
                    // OpenPose output format to cv::Mat
                    outputImage = opOutputToCvMat.formatToCvMat(outputArray);
                } else {
                    // Handle case when no user flag is set. Just copy input to output
                    outputImage = inputImage;
                }
            }

            // Get l2/cosine distance samples for poseKeypoints
            if (FLAGS_pose) {
                // Pose Parameters

                const auto &pose_pairs = POSE_BODY_PART_PAIRS_RENDER[(int) poseModel];

                pm.getSamplesFromPoseKeypoints(outputImage, m_pose_sample, poseKeypoints, pose_label_count,
                                               pose_pairs, pose_label,
                                               (float) FLAGS_pose_render_threshold, pose_names,
                                               frame_count_per_sample, pose_count,
                                               pose_predicted_class_str, pose_predicted_score_str,
                                               (float) FLAGS_pose_classifier_threshold,
                                               pose_labels, pose_samples, generate_pose, true, FLAGS_debug,
                                               k_pose_model);
            }

            // Get l2/cosine distance samples for handKeypoints
            if (FLAGS_hand) {
                // Hand Parameters
                const auto &hand_pairs = HAND_PAIRS_RENDER;

                // Left Hand
                pm.getSamplesFromHandKeypoints(outputImage, m_left_hand_sample, handKeypoints[0],
                                               hand_label_count, hand_pairs, hand_label,
                                               (float) FLAGS_hand_render_threshold, hand_names,
                                               frame_count_per_sample, hand_count,
                                               left_hand_predicted_class_str, left_hand_predicted_score_str,
                                               (float) FLAGS_hand_classifier_threshold,
                                               left_hand_labels, left_hand_samples, KP_LEFT_HAND, generate_hand,
                                               false, FLAGS_debug, k_hand_model);
                // Right Hand
                pm.getSamplesFromHandKeypoints(outputImage, m_right_hand_sample, handKeypoints[1],
                                               hand_label_count, hand_pairs, hand_label,
                                               (float) FLAGS_hand_render_threshold, hand_names,
                                               frame_count_per_sample, hand_count,
                                               right_hand_predicted_class_str, right_hand_predicted_score_str,
                                               (float) FLAGS_hand_classifier_threshold,
                                               right_hand_labels, right_hand_samples, KP_RIGHT_HAND,
                                               generate_hand, false, FLAGS_debug, k_hand_model);
            }

            // Get l2/cosine distance samples for faceKeypoints
            if (FLAGS_face) {

                // Face Parameters
                const auto &face_pairs = FACE_PAIRS_RENDER;

                pm.getSamplesFromFaceKeypoints(outputImage, m_face_sample, faceKeypoints, face_label_count,
                                               face_pairs, face_label,
                                               (float) FLAGS_face_render_threshold, face_names,
                                               frame_count_per_sample, face_count,
                                               face_predicted_class_str, face_predicted_score_str,
                                               (float) FLAGS_face_classifier_threshold,
                                               face_labels, face_samples, generate_face, false, FLAGS_debug,
                                               k_face_model);
            }

            // Check for frame count per sample
            if (frame_count_per_sample == timesteps - 1)
                frame_count_per_sample = 0;
            else
                frame_count_per_sample++;

        }

        // copy classified/rendered image for display
        origImage = outputImage;

        // ------------------------- SHOWING RESULT AND CLOSING -------------------------
        cv::imshow("pose image",origImage);

        // record outputImage frames as video file
        if (FLAGS_record)
            outputVideo << origImage;

    } // end of while

    // release opencv matrix resource
    inputImage.release();
    outputImage.release();
    origImage.release();
    cv::destroyAllWindows();

    cout << "Finished successfully..!!" << endl;

    // Logging information message
    op::log("openpose pose, gesture, emotion and dlib face recognition successfully finished.", op::Priority::High);

    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseRecognition");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseRecognition
    return openPoseRecognition();
}


