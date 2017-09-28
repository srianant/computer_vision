# "Recognition" Design:

The purpose of this work is to demonstrate few state of art computer vision applications like Gesture, Emotions, Posture and Face Recognitions using popular computer vision open source libraries OpenPose/DLIB.

**This work is greatly inspired and have used code, concepts from libraries like OpenPose, DLIB, Kerasify and OpenCV.**

The **recognition** application design and code is rightly integrated into OpenPose C++/Multithreaded framework. Fig(1) shows software architecture. Gesture, Emotions and Posture recognition is built based on OpenPose hand, face and pose libraries respectively. While Face recognition is built based on DLIB computer vision libraries.


Fig(1): Software Architecture    
Modules in orange are newly added/integrated for "recognition". Grey modules are of native OpenPose.    
<img src="images/software_arch.png"  height="400"/>   

Fig(1a): OpenPose Architecture    [[1]](https://arxiv.org/pdf/1611.08050.pdf)  
<img src="images/openpose_arch.png" height="400"/>   


## Classifier:

A Time Distributed Feed Forward (Dense) neural network with LSTM classifier is used to classify openpose samples. Fig(1a) shows Keras Classifier Model. The model is trained with various 2D keypoints estimates with certain distance metric.

Consider a batch of 32 samples, where each sample is a sequence of 5 vectors of 36 dimensions. The batch input shape of the layer is then (32, 5, 36), and the input_shape, not including the samples dimension, is  (5, 36). Fig(1a) depicts network for posture(pose) with sample vector dimension of 36. Details of sample vector and distance metric are explained in below sections.  

Fig(1b): Keras Classifier Model  
<img src="images/keras_network.png" height="600"/>


## OpenPose:  
A realtime multi-person skeletal 2D pose estimation deep neural network that locates anatomical keypoints for each person body parts such as limbs, hand, leg etc. using part affinity fields. Details of their design and research can be found [here](https://arxiv.org/pdf/1611.08050.pdf).  

### Gesture Recognition:  
OpenPose **Hand Keypoints** illustrated in Fig(2) is used to classify different human hand gestures like victory, wave, stop, fist etc. The **recognition** application constructs a sample vector using cosine distance measured from reference keypoint (0: wrist) to all other hand keypoints as show in the Fig(2). This distance metric allows us to uniquely classify different gestures.  

Fig(2): OpenPose Hand Keypoints   [[2]](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)  
<img src="images/keypoints_hand.png" height="300"/>   

Here are few hand keypoints rendered test samples with prediction and its confidence score.

Fig(3a): **Victory**  
<img src="images/victory.png" height="200"/>

Fig(3b): **Stop**  
<img src="images/stop.png" height="200"/>

Fig(3c): **ThumbsUP**  
<img src="images/thumbsup.png" height="200"/>   

Fig(3d): **Pinch**  
<img src="images/pinch.png" height="200"/>

#### Gesture classifier training:
**Model parameters:**  
epochs      = 1000  
timesteps   = 5    
batch_size  = 32  
dropout     = 0.1  
activation  ='relu'  
optimizer   ='Adam'   
vector_dim  = 40    

**Confusion Matrix and Validation Accuracy:**  
Validation Accuracy: 97%  
Number of samples: < 600  
<img src="images/hand_training.png" height="250"/>

### Emotions Recognition:  
OpenPose **Face Keypoints** illustrated in Fig(4) is used to classify different human face emotions like sad, happy, surprise and normal. The **recognition** application constructs a sample vector using both l2 and cosine distance measured from reference keypoint (30: tip of nose) to all other face keypoints as show in Fig(4). To reduce the high dimensionality of vector space and better classify keypoints 0 thru 16 (chin and jaw) are ignored. Combining both distance metrics allows us to uniquely classify different face emotions.  


Fig(4): OpenPose Face Keypoints   [[1]](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)  
<img src="images/keypoints_face.png" height="300"/>   

Here are few face emotions rendered test samples with prediction and its confidence score.

Fig(4a): **Sad**  
<img src="images/sad.png" height="200"/>

Fig(4b): **Happy**  
<img src="images/happy.png" height="200"/>

Fig(4c): **Surprise**  
<img src="images/surprise.png" height="200"/>   

Fig(4d): **Normal**  
<img src="images/normal.png" height="200"/>

#### Emotions classifier training:
**Model parameters:**  
epochs      = 1000  
timesteps   = 5    
batch_size  = 32  
dropout     = 0.1  
activation  ='tanh'  
optimizer   ='Adadelta'   
vector_dim  = 96    

**Confusion Matrix and Validation Accuracy:**  
Validation Accuracy: 96%   
Number of samples: < 500  
<img src="images/face_training.png" height="250"/>  

### Posture Recognition:  
OpenPose **Pose Keypoints** illustrated in Fig(5) is used to classify different human pose (posture) like sitting, standing and close_to_camera. The **recognition** application constructs a sample vector using l2 distance measured from reference keypoint (0: neck) to all other pose keypoints as show in Fig(5). This distance metric allows us to uniquely classify different human posture.  

Fig(5): OpenPose Pose Keypoints   [[1]](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)  
<img src="images/keypoints_pose.png" height="300"/>    

Here are few pose rendered test samples with prediction and its confidence score.

Fig(5a): **Sitting**  
<img src="images/sitting.png" height="200"/>

Fig(5b): **Standing**  
<img src="images/standing.png" height="200"/>

Fig(5c): **Close_to_camera**  
<img src="images/close_to_camera.png" height="200"/>   

#### Posture classifier training:
**Model parameters:**  
epochs      = 1000  
timesteps   = 5    
batch_size  = 32  
dropout     = 0.1  
activation  ='relu'  
optimizer   ='Adam'   
vector_dim  = 36    

**Confusion Matrix and Validation Accuracy:**  
Validation Accuracy: 98%  
Number of samples: < 400  
<img src="images/pose_training.png" height="250"/>


## DLIB:
A modern C++ toolkit containing deep learning algorithms and tools for creating computer vision software. Details of DLIB library and supported features can be found [here](https://github.com/davisking/dlib).  

### Face Recognition:  
Face rectangle detected from OpenPose face keypoints is used for DLIB face recognition. Our observation found OpenPose face detection is much faster as its CNN than DLIB HOG based detection models.

Following steps are applied to recognize/train faces:  
1) With ROI of face rectangle identify 68 facial landmarks using DLIB shape detector. These landmarks are in same positions as OpenPose, except for two addition of landmarks (left and right) eye balls. The iBUG 300-W face landmark annotation scheme is followed.  

2) Affine transformation: Face will be rotated upright, centered, and scaled.  

3) Call DNN (ResNet based) to convert each face image in faces into a 128D vector. These 128D vector is unique to each face.

## Limitations:

### OpenPose Recognitions:
- Samples are trained with one person for this proof of concept and modeled accordingly. The respective samples and Keras model can be found in train_data folder. Though limited testing was done with more than one person.  

- Since the classifier design is based on distance metric it would make more sense to have training data sampled from multiple people with different hand, body and face features.  


### DLIB Face Recognition:
- Our experiments found ResNet DNN model works fine for most cases, but unable to recognize face when it is completely turned right or left, pointing upwards or downwards.  

## Reference:
[[1]](https://arxiv.org/pdf/1611.08050.pdf) Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields  
[[2]](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md) OpenPose: A Real-Time Multi-Person Keypoint Detection And Multi-Threading C++ Library  
[[3]](https://github.com/davisking/dlib) DLIB: A toolkit for making real world machine learning and data analysis applications in C++  
[[4]](https://github.com/moof2k/kerasify) Small library for running Keras models from a C++ application  
