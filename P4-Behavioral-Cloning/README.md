# Project 4 - Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Track 1             |  Track 2
:-------------------------:|:-------------------------:
![](./track1.gif)  |  ![](./track2.gif)

Overview
---

In this project, were used a convolutional neural network to clone driving behavior using imitation learning. The goal was to create an agent capable of navigate correctly at least in track 1, no tire leaving the drivable portion of the track surface. 

Multiple experiments with different architectures of convolutional neural networks were made, like for example, with the one proposed on [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)), and the architecture with the best performance was [EfficientNetB0](https://arxiv.org/pdf/1905.11946.pdf). The neural network use as input an image from a front-facing camera and outputs a steering angle to maintain the vehicle on track. To control the speed of the vehicle, it's used a PI controller with a set speed of 20 mph.

The final results can be seen on youtube using the links below:
* Track1: https://youtu.be/XZHiomgoXMc
* Track2: https://youtu.be/BuaSlohBVjY

The Project
---
The steps of this project was the following:
* Use the simulator to collect data of good driving behavior: Were collected data from track1 and track2, driving clockwise and counter-clockwise, always trying to keep it in the center of the lane. 
* Data preprocessing: Images from left and right camera were used to augment data. For left images, an offset of 0.2 were added to steering angles, and for right images, an offset of -0.2 were added to steering angles. Those offsets is necessary since left and right cameras are displaced in relation to the central camera, which is used later to drive the car. For full details see "1. Data preprocessing".
* Design, train and validate a model that predicts a steering angle from image data: Diffent models were used but EfficientNetB0 were the most efficient (couldn't miss the joke). For more details of which parameters were used for the network, see "2. Train Model".
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file (keras model file), i.e. `model.h5`. It can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
