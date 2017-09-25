/*
 * File Description :   File defines DLIB model class object functions for DLIB face recognition.
 *                      Face rectangle detected from openpose is used for dlib face recognition.
 * File Name        :   dlib_model.cpp
 * Author           :   Srini Ananthakrishnan
 * Date             :   09/23/2017
 * Github           :   https://github.com/srianant
 * Reference        :   DLIB c++ examples (https://github.com/davisking/dlib)
 */

#include <openpose/user_code/dlib_model.hpp>

// Read filenames from train_data/dlib_faces.txt
void dlib_model::readTrainFilenames( std::string& filename, std::string& dirName, std::vector<string>& trainFilenames)
{
    trainFilenames.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    size_t pos = filename.rfind('\\');
    char dlmtr = '\\';
    if (pos == string::npos)
    {
        pos = filename.rfind('/');
        dlmtr = '/';
    }
    dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        trainFilenames.push_back(str);
    }
    file.close();
}

// Read standard input with timeout
std::string dlib_model::readStdIn(int timeout)
{
    struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };

    std::string line="";
    int ret = 0;
    while(ret == 0)
    {
        ret = poll(&pfd, 1, timeout*1000); // timeout in msec
        if(ret == 1) // there is something to read
        {
            std::getline(std::cin, line);
        }
        else if(ret == -1)
        {
            std::cout << "Error: " << strerror(errno) << std::endl;
        }
        break;
    }
    return line;
}

// Train Faces API based on DLIB (openface) library
// Steps for training
// 1) Serialize pre-trainned weights for shape detector 68 face landmarks
// 2) Serialize pre-trainned weights (Tourch based NN - Resnet) for face recognition
// 3) Loop thru all images in a directory and stack them in matrix
// 4) Use DLIB face detector to crop the faces from the image and display
// 5) Cropped face is aligned (rotated and scaled)
// 6) 128D feature discriptor (NN) extracted
// 7) Train Faces feature discriptor are stored
int dlib_model::facerec_dlib_train_face_images(int initialize_done)
{
    string trainDirName;
    std::vector<string> trainImageNames;
    string trainFilename = "train_data/dlib_faces.txt";
    dlib::image_window win, win_faces;
    int start_image = 0;

    if(initialize_done == 0)
    {
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        if (std::ifstream("/home/image/dlib/shape_predictor_68_face_landmarks.dat"))
            deserialize("/home/image/dlib/shape_predictor_68_face_landmarks.dat") >> sp;
        else
        {
            cout << "ERROR: DLIB shape predictor 68 face landmarks dat file not found" << endl;
            return (0);
        }

        // And finally we load the DNN responsible for face recognition.
        if (std::ifstream("/home/image/dlib/dlib_face_recognition_resnet_model_v1.dat"))
            deserialize("/home/image/dlib/dlib_face_recognition_resnet_model_v1.dat") >> net;
        else
        {
            cout << "ERROR: DLIB face recognition resnet model v1 dat file not found" << endl;
            return (0);
        }
    }


    readTrainFilenames( trainFilename, trainDirName, trainImageNames );
    if( trainImageNames.empty() )
    {
        cout << "Train image filenames can not be read." << endl << ">" << endl;
        return (0);
    }

    if(initialize_done > 0)
        start_image = train_face_descriptors.size();

    for( size_t i = start_image; i < trainImageNames.size(); i++ )
    {
        string filename = trainDirName + trainImageNames[i];

        matrix<rgb_pixel> img;
        if (std::ifstream(filename))
            load_image(img, filename);
        else{
            cout << "Unable to find file:" << filename << endl;
            continue;
        }
        std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string trainImageName = base_filename.substr(0, p);

        dlib_face_names.push_back(trainImageName);
        // Make the image larger so we can detect small faces.
        pyramid_up(img);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<dlib::rectangle> dets = detector(img);

        if (dets.size() == 0)
        {
            cout << "skipping image file:" << filename << " no face detected"<< endl;
            continue;
        }
        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected and store them in shape.
        auto shape = sp(img, dets[0]);

        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        matrix<rgb_pixel> train_face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), train_face_chip);
        train_faces.push_back(move(train_face_chip));
    }

    if (train_faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }
    cout << "Total Faces trained:" << train_faces.size() << endl;

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.
    //double cpu_t = cv::getTickCount();
    train_face_descriptors = net(train_faces);
    //double nnTrainTime = (cv::getTickCount() - cpu_t)/cv::getTickFrequency();
    //cout << "DNN ALL faces to 128D vector space train time: " << nnTrainTime << " ms" << endl;

    for (size_t i = 0; i < train_face_descriptors.size(); ++i)
    {
        for (size_t j = i+1; j < train_face_descriptors.size(); ++j)
        {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (length(train_face_descriptors[i]-train_face_descriptors[j]) < 0.6)
                cout << "Training: Found matching in training faces :" << i << " and " << j << endl;
        }
    }
    return 1;
}

// Recognize face using following algorithm.
// FaceRec Algo (Openpose + DLIB):
// 1) Extract face keypoints and Bounding rectangle from openpose
// 2) Populate DLIB class full_object_detection i.e shape with values from step-1
// 3) DLIB extract_image_chip() and obtain test/train faces
// 4) Call DNN to convert each face image in faces into a 128D vector. net(test_faces)
// 5) Obtain face descriptor and classify
void dlib_model::dlib_facerec_from_faceRectangles(cv::Mat &_outputImage,const Array<float>& _faceKeypoints,
                                      int & _frame_count_per_sample, bool *_face_capture, bool _debug)
{
    // Parameters
    const auto thresholdRectangle = 0.1f;
    const auto numberKeypoints = _faceKeypoints.getSize(1);

    for (auto person = 0 ; person < _faceKeypoints.getSize(0) ; person++)
    {
        int best_class = 0; // best match class
        double len = 0.0;
        double best_len = 9999.0;
        size_t i, j;
        std::vector<matrix<rgb_pixel>> test_faces;
        std::vector<matrix<float,0,1>> test_face_descriptors;

        const auto personRectangle = op::getKeypointsRectangle(_faceKeypoints, person, numberKeypoints, thresholdRectangle);
        if (personRectangle.area() > 0)
        {
            Mat frame;
            cvtColor(_outputImage, frame, CV_BGR2GRAY);
            // A.left(), A. top(), A.right() and A.bottom() for x, y, w, and h respectively.
            // Crop the original image to the defined ROI
            cv::Rect roi;
            roi.x = intRound(personRectangle.x) - intRound(personRectangle.height)/2;
            roi.y = intRound(personRectangle.y) - intRound(personRectangle.width)/2;
            roi.width = intRound(personRectangle.width) + intRound(personRectangle.x)/2;
            roi.height = intRound(personRectangle.height) + intRound(personRectangle.y)/3;

            // Under certain conditions as below ignore the frame processing to avoid openCV assertion
            if (roi.x + roi.width > frame.cols || roi.x < 0 || roi.width < 0)
                continue;

            if (roi.y + roi.height > frame.rows || roi.y < 0 || roi.height < 0)
                continue;


            cv::Mat queryImage;
            try {
                queryImage = frame(roi);
            }
            catch (const std::exception& e) {
                cout << "Error..!!" << endl;
                continue;
            }

            dlib::array2d<unsigned char> img;
            assign_image(img, cv_image<unsigned char>(queryImage));

            // Make the image larger so we can detect small faces.
            pyramid_up(img);

            std::vector<dlib::rectangle> dets = detector(img);
            if (dets.size() == 0)
            {
                if(_debug) // # represent frame unable to detect face
                    cout << "#";
                continue;
            }
            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected and store them in shape.
            // This object is a tool that takes in an image region containing some object and outputs a set of
            // point locations that define the pose of the object. The classic example of this is human face pose
            // prediction, where you take an image of a human face as input and are expected to identify the
            // locations of important facial landmarks such as the corners of the mouth and eyes, tip of the nose,
            // and so forth.
            auto shape = sp(img, dets[0]);

            matrix<rgb_pixel> test_face_chip;
            // get_face_chip_details(): Affine transformation
            // This function assumes det contains a human face detection with face parts
            // annotated using the annotation scheme from the iBUG 300-W face landmark
            // dataset.  Given these assumptions, it creates a chip_details object that will
            // extract a copy of the face that has been rotated upright, centered, and
            // scaled to a standard size when given to extract_image_chip().
            // The extracted chips will have size rows and columns in them.
            // if padding == 0 then the chip will be closely cropped around the face.
            // Setting larger padding values will result a looser cropping.  In particular,
            // a padding of 0.5 would double the width of the cropped area, a value of 1
            // would triple it, and so forth.

            // extract_image_chip():
            // This function extracts "chips" from an image. That is, it takes a list of rectangular
            // sub-windows (i.e. chips) within an image and extracts those sub-windows, storing each
            // into its own image. It also allows the user to specify the scale and rotation for the chip.
            extract_image_chip(img, get_face_chip_details(shape,150,0.25), test_face_chip);
            test_faces.push_back(move(test_face_chip));

            //double cpu_t = cv::getTickCount();
            test_face_descriptors = net(test_faces);
            //double nnTrainTime = (cv::getTickCount() - cpu_t)/cv::getTickFrequency();
            //cout << "TEST: DNN ALL faces to 128D vector space train time: " << nnTrainTime << " ms" << endl;

            for (i = 0; i < test_face_descriptors.size(); ++i)
            {
                for (j = 0; j < train_face_descriptors.size(); ++j)
                {
                    // Faces are connected in the graph if they are close enough.  Here we check if
                    // the distance between two face descriptors is less than 0.6, which is the
                    // decision threshold the network was trained to use.  Although you can
                    // certainly use any other threshold you find useful.
                    len = length(test_face_descriptors[i]-train_face_descriptors[j]);
                    if (len < 0.5)
                    {
                        //cout << "TEST: Found matching training face:"  << j+1 << " with length:" << len << endl;
                        if (len < best_len)
                        {
                            best_len = len;
                            best_class = j;
                        }
                    }
                }
            }

            if (test_faces.size() == 0)
            {
                cout << "No faces found in image!" << endl;
                continue;
            }

            if (best_len != 9999.0)
            {
                if(_frame_count_per_sample == timesteps-1)
                {
                    if(_debug)
                        cout << "Predicted face:" << person << " Name:" << dlib_face_names[best_class].c_str() << endl;
                }

                string box_text = format("Face = %s ", dlib_face_names[best_class].c_str());
                // And now put it into the image:
                putText(_outputImage, box_text, cv::Point(personRectangle.x, personRectangle.y-20),
                        FONT_HERSHEY_PLAIN, 1.25, CV_RGB(0,255,255), 2.0);
            }
            else
            {
                if(*_face_capture)
                {
                    cout << endl << "Looks like we found a new face...lets add it!!!:" << endl;
                    cv::imshow("new face",queryImage);

                    cv::waitKey(5);
                    cout << "Please enter a name:" << endl;
                    std::string name = readStdIn(5); // check user input for 5 sec
                    if(!name.empty())
                    {
                        dlib_face_names.push_back(name);
                        train_face_descriptors.push_back(test_face_descriptors[0]);
                    }
                    *_face_capture = false;
                    cout << "Capturing new face...: OFF" << endl;


                }
            }
        }
    } // end of person for face
}

