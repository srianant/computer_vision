
# Gesture, Emotions, Posture and Face Recognition using OpenPose/DLIB
Purpose of this work is to demonstrate few state of art computer vision applications using OpenPose/DLIB libraries.

## Acknowledgements:
This github repository work is greatly inspired and have used code, concepts presented in the following github repositories:

- [DLIB](https://github.com/davisking/dlib):  Modern C++ toolkit for computer vision and other machine learning.  
- [Kerasify](https://github.com/moof2k/kerasify) : Small library for running Keras models from a C++ application.  
- [OPENCV](https://github.com/opencv/opencv): Open Source Computer Vision Library.  
- [OPENPOSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose): A Real-Time Multi-Person Keypoint Detection And Multi-Threading C++ Library.  

## Operating systems:
- Ubuntu 16.04
- [Nvidia Jetson TX2](https://developer.nvidia.com/embedded/buy/jetson-tx2)  

## Requirements:

- NVIDIA graphics card with at least 1.6 GB available (the nvidia-smi command checks the available GPU memory in Ubuntu).  
- At least 2 GB of free RAM memory.
- Highly recommended: cuDNN and a CPU with at least 8 cores.

## Installation and Compiling:

- [install_packages](https://github.com/srianant/computer_vision/blob/master/openpose/installation.md)   (INSTALLATION IS MUST..)
- copy following directories/files from this github repo to respective openpose folders
  - copy from_this_repo/openpose/Makefile ~openpose/Makefile
  - copy from_this_repo/openpose/Makefile.config ~openpose/Makefile.config
  - copy from_this_repo/openpose/3rdparty/caffe/Makefile ~openpose/3rdparty/caffe/Makefile
  - copy from_this_repo/openpose/3rdparty/caffe/Makefile.config ~openpose/3rdparty/caffe/Makefile.config
  - copy -r from_this_repo/openpose/train_data ~/openpose/
  - copy -r from_this_repo/openpose/examples/user_code ~/openpose/examples
  - copy -r from_this_repo/openpose/src/openpose/user_code ~/openpose/src/openpose
  - copy -r from_this_repo/openpose/inc/openpose/user_code ~/openpose/inc/openpose  


- From ~/openpose/3rdparty/caffe/ directory  
  (Make sure you have compiled openpose and executed few examples)
> make clean  
> make -j8  

- From ~/openpose/ directory  
> make clean  
> make -j8

## How to Run:
(From openpose root directory ~/openpose/)  
- Gesture Recognition:
>./build/examples/user_code/openpose_recognition.bin --hand  

- Emotions Recognition:  
>./build/examples/user_code/openpose_recognition.bin --face  

- Pose Recognition:  
>./build/examples/user_code/openpose_recognition.bin --pose  

- Face Recognition:  
>./build/examples/user_code/openpose_recognition.bin --dlib_face  

You could also run following combinations of recognitions:
> ./build/examples/user_code/openpose_recognition.bin --hand --pose  
> ./build/examples/user_code/openpose_recognition.bin --face --pose   
> ./build/examples/user_code/openpose_recognition.bin --dlib_face --pose  
> ./build/examples/user_code/openpose_recognition.bin --dlib_face --face   
  
WARNING: Due to memory constraint FACE and HAND recognitions are not allowed simultaneously..!!. To run this you need minimum of 2x GPU memory requirements.

## Generating New Keras Classifier model:
This repository comes with experimented samples, pre-compiled classifier models and can be found in from_this_repo/openpose/train_data folder.

After installation and compiling steps:  
(Note: virtualenv named "tensorflow" was used in my environment to install tensorflow)  
> source tensorflow/bin/activate  
(tensorflow) ubuntu:$ python openpose/examples/user_code/python/rnn_lstm_classifier.py

Note: Jupyter notebook version of above python file can also be found in same directory. Recommended to use if you wish to perform model visualizations

## Recognitions:

### Gesture recognition:

<img src="output/hand_gesture_video.gif" height="400"/>

### Emotions recognition:

<img src="output/emotions_video.gif" height="400"/>

### Pose recognition:

<img src="output/pose_video.gif" height="400"/>

### DLIB Face recognition:

<img src="output/dlib_face_recognition.gif" height="400"/>
