# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataset_example_images_steer.png "Dataset Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `1. Data preprocessing.ipynb` containing the code to preprocess the dataset and split data into training and validation. 
* `2. Train Model.ipynb` containing script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

Notebooks `1. Data preprocessing.ipynb` and `2. Train Model.ipynb` contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a modified version of EfficientNet B0. It uses the convolutional layers of EfficientNet, and in the end I removed it's fully connected layer, and added a dropout and a fully connected layer.

The model includes a preprocessing function and expects an image with float pixels values in the range of [0-255]. I choose to use this network because I was aiming to create a model capable of drive on track1 and track2, and in my tests, the proposed architecture from [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) and my custom made networks wasn't performing well on track2. [EfficientNet](https://arxiv.org/abs/1905.11946) is a really powerfull and lightweight network, and in it's original paper, it's shown that the proposed architecture outperforms all previous state of the art networks in ImageNet. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer, with 25% of probability, in order to reduce overfitting (2. Train Model.ipynb cell 5, line 11). 

The model was trained using a early stopping scheduler to avoid overfitting, and was used the weights from the last epoch with an improvement in the loss. Finally, it was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, a learning rate of 0.0001/0.00001 and a batch size of 32. The training consists of 2 steps, first training the newly added custom layers, with efficientNet layers frozen and a learning rate of 0.0001, then, training the entire network with efficientNet layers unfrozen and a smaller learning rate of 0.00001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Were collected data from track1 and track2, driving clockwise and counter-clockwise, always trying to keep it in the center of the lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning, using a state of the art model from ImageNet competition. 

My first step was to use a convolution neural network model similar to the one from [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) I thought this model might be appropriate because it was already used for behaviour cloning, so I started from there. But as I said before, although this model drives well on track 1, it can't perform well on track2, So after trying some variations and other custom made networks, I decided to try transfer learning with EfficientNetB0 model, which worked really well.

To evaluate my models performance I used the mean squared error of each epoch of training. After founding a model with at least 0.05 of mse, I tested it directly on the simulator and used driving behaviour as an indication of success or fail.

To speed up the training and help the model to generalize better, I removed parts of the image from the camera, like the sky and the car hood, leaving almost only the track.

In the beginning of the training, I realized that the car was always driving too much to the right. To undestand what was happened I plotted the data distribution and discovered that most of my data was from examples turning to the right, with a steering angle of 0.2. To get rid of this problem, I decided to augment my data using a flipped copy of each image, with the inverse steering angle. This helped the car to drives for a while, but in "hard" curves, it was always going off the limits of track. To solve this problem I added more examples of the car driving in curves and started to use images from left and right camera, adding an offset to their corresponding steering angles. The offset is added as a way to compensate for the fact that they are offset in relation to the car's central camera, that is, seeing from the perspective of these cameras, the steering angle of the car is different from the one originally collected, it must be offset. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (2. Train Model.ipynb cell 8) consisted of EfficientNetB0 convolutional layers working as feature extractor, and a dropout followed by a dense layer to predict the steering angle.

Here is a visualization of the architecture, according to keras summary function:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 76, 320, 3)]      0         
_________________________________________________________________
efficientnetb0 (Functional)  (None, 3, 10, 1280)       4049571   
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1280)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 1281      
=================================================================
```

#### 3. Creation of the Training Set & Training Process

To create the training set I recorded 5 laps using center lane driving clockwise, and 3 laps counter-clockwise on track1 and track2. I also recorded some recovery laps, from the car driving in curves on track1 to give the model more examples of driving in those parts of the track. 

With those laps recorded, I removed from the images, pixels with information from the sky and the car hood, leaving only information from track in the images. I also added an offset to left and right cameras steering angles, to compensate their offset from the center camera. Below it's shown some examples of images and steering angles from the dataset:

![image1]

To solve database imbalance, I added a flipped version of all images with inverted the steering angle (-steering_angle) to the dataset. Finally I randomly shuffled data, splitted it into training and validation set, and saved both sets in a Numpy binary file format (.npy) to speed up the training of multiple models later.