
## Installation Steps: (on Ubuntu 16.04)

#### Update system package index
> sudo apt update;

#### Install OS libraries (cmake, atlas, boost)  
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

#### Install OpenPose  
[openpose_install](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md)  

Improtant: Make sure to run openpose suggested examples.

Congrats..!!! You have completed installing all necessary packages.  
