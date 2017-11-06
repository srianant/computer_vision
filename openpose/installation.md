
## Installation Steps: (on Ubuntu 16.04)

#### Update system package index
> sudo apt update;

#### Install OS libraries (cmake, atlas, boost(1.58.0.1))  
> sudo apt-get install build-essential cmake pkg-config  
> sudo apt-get install libx11-dev libatlas-base-dev  
> sudo apt-get install libgtk-3-dev libboost-python-dev  
> sudo apt-get install libboost-all-dev  

#### Install Python Libraries
> sudo apt-get install python-dev python-pip python3-dev python3-pip  
> sudo -H pip2 install -U pip numpy  
> sudo -H pip3 install -U pip numpy  
> pip install numpy scipy matplotlib scikit-image scikit-learn ipython

#### Install CUDA, cUDNN, TensorFlow, Keras, H5Py
https://github.com/lwneal/install-keras

#### Install OpenCV  
> sudo apt-get install libopencv-dev python-opencv  

Download the installation script [install-opencv.sh](https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh), open your terminal and execute:  

#### Install DLIB as Library
> wget <a class="vglnk" href="http://dlib.net/files/dlib-19.6.tar.bz2" rel="nofollow"><span>http</span><span>://</span><span>dlib</span><span>.</span><span>net</span><span>/</span><span>files</span><span>/</span><span>dlib</span><span>-</span><span>19</span><span>.</span><span>6</span><span>.</span><span>tar</span><span>.</span><span>bz2</span></a>  
> tar xvf dlib-19.6.tar.bz2  
> cd dlib-19.6/  
> mkdir build  
> cd build  
> cmake ..  
> cmake --build . --config Release  
> sudo make install  
> sudo ldconfig  
> cd ..  

#### Install OpenPose (version: 1.0.0)  
[openpose_install](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)  

Improtant: Make sure to run openpose suggested examples.

Congrats..!!! You have completed installing all necessary packages.  

## Compiling  

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
