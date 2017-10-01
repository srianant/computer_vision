
# Gesture, Emotions, Posture and Face Recognition using OpenPose/DLIB
Purpose of this work is to demonstrate few state of art computer vision applications using OpenPose/DLIB libraries.

## Acknowledgements:
This github repository work is greatly inspired and have used code, concepts presented in the following github repositories:

- [DLIB](https://github.com/davisking/dlib):  Modern C++ toolkit for computer vision and other machine learning.  
- [Kerasify](https://github.com/moof2k/kerasify) : Small library for running Keras models from a C++ application.  
- [OPENCV](https://github.com/opencv/opencv): Open Source Computer Vision Library.  
- [OPENPOSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose): A Real-Time Multi-Person Keypoint Detection And Multi-Threading C++ Library.  

Thanks to Dr.Michael Rinehart, Chief Scientist at Elastica for his mentorship and guidance through the project.   


## Operating systems (supported):
- Ubuntu 16.04
- [Nvidia Jetson TX2](https://developer.nvidia.com/embedded/buy/jetson-tx2)  


## Requirements:

- NVIDIA graphics card with at least 1.6 GB available (the nvidia-smi command checks the available GPU memory in Ubuntu).  
- At least 2 GB of free RAM memory.
- Highly recommended: cuDNN and a CPU with at least 8 cores.

## Install, Compile and Run:

- [install_compile_and_run](https://github.com/srianant/computer_vision/blob/master/openpose/installation.md)   (INSTALLATION IS MUST..)

## Design:  
- [Software design](https://github.com/srianant/computer_vision/blob/master/openpose/Readme.md)    

## Demo:  

### Gesture recognition:  

<img src="output/hand_gesture_video.gif" height="400"/>  

### Emotions recognition:  

<img src="output/emotions_video.gif" height="400"/>  

### Pose recognition:  

<img src="output/pose_video.gif" height="400"/>  

### DLIB Face recognition:  

<img src="output/face_rec.gif" height="400"/>  
