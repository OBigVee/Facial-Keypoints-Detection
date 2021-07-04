[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# -Facial-Key-Points-Detection-System

## Project Overview
in this project i combined the knowlege of computer vision techniques and deep learning architectures to build a facial keypoint detection system.
detections of keypoints in any provided faces such as nose, mouth, eyes on faces and this is used in many applications including facial pose recognition, facial filters and emotion recognition. this code is able to look at any image, detect faces and predict the locations of facial keypoints on face, example of this is the below image

![Facial Keypoint Detection][image1]


## Dependencies
### Set up your Enviroment
1. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

2. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```
make sure all these packages are installed
```
opencv-python==3.2.0.6
matplotlib==2.1.1
pandas==0.22.0
numpy==1.12.1
pillow==5.0.0
scipy==1.0.0
torch>=0.4.0
torchvision>=0.2.0
```

## Models.py 
this file contains the CNN architecture i designed for this project

## data_load.py
this file contains some functions use to transform the data

## dectector_architectures
this folder has some haarcascade files in it

# First_Models_Saved
this folder contains my model checkpoint so i wouldn't need to train the neural network i'd just load the saved checkpoint
