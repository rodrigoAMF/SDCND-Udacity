# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution_train_set.png "Train set distribution"
[image2]: ./examples/distribution_validation_set.png "Validation set distribution"
[image3]: ./examples/distribution_test_set.png "Test set distribution"
[image4]: ./examples/images_from_dataset.png "Example of images from the dataset"
[image5]: ./test_images/7_speed_limit_100.jpg "Traffic Sign 1"
[image6]: ./test_images/13_yield.jpg "Traffic Sign 2"
[image7]: ./test_images/17_noentry.jpg "Traffic Sign 3"
[image8]: ./test_images/31_wildanimalscrossing.jpg "Traffic Sign 4"
[image9]: ./test_images/34_turn_left_ahead.jpg "Traffic Sign 5"


[image5]: ./examples/grayscale.jpg "Grayscaling"
[image6]: ./examples/random_noise.jpg "Random Noise"
[image7]: ./examples/placeholder.png "Traffic Sign 1"
[image8]: ./examples/placeholder.png "Traffic Sign 2"
[image9]: ./examples/placeholder.png "Traffic Sign 3"
[image10]: ./examples/placeholder.png "Traffic Sign 4"
[image11]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rodrigoAMF/SDCND-Udacity/tree/master/P3-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used only python built-in libraries and numpy to calculate get info of the traffic signs dataset:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I also used matplotlib to plot some images from the dataset:

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to use RGB images in order to improve the performace of the algorithm. 

For the preprocessing step, I only applied a normalization for the pixels, by doing `pixel = pixel/255`. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					          | 
|:---------------------:|:-------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							          |
| Convolution 5x5     	| 1x1 stride, 6 filters, valid padding, outputs 28x28x6   |
| RELU					|												          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				              |
| Convolution 5x5	    | 1x1 stride, 6 filters, valid padding, outputs 10x10x16  |
| RELU					|												          |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				              |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x120              |
| RELU					|												          |
| Fully connected		| 84 neurons        						              |
| RELU					|												          |
| Fully connected		| 43 neurons       							              |
| Softmax				| Probability of each class          				      |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, was used adam as optimizer, with a starting learning rate of 0.002, that decays 0.5 every 5 epochs without an improvement on validation loss until a minimum of 0.0005. The decay of the learning rate was usefull to prevent overtting in the final epochs of the training.

Only the model with the best performance, where performance is evaluated by looking at validation accuracy, was saved (at models folder with the name of best_model.h5). Also, if there is no improvement on the validation loss during 10 consecutives epochs, the training stops. 

The size of the batch used to update the weights of the network was 32, and the best model trained 12 epochs.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9980
* validation set accuracy of 0.9510
* test set accuracy of 0.9336

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
LeNet-5
* What were some problems with the initial architecture?
It wasn't good enough to get an accuracy on the validation set over 0.93.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I changed the activation function used between the convolutional and fully connected layers to Relu, since it gives a more stable training. I also changed the layer of average pooling to max pooling, because in my tests the network performed better with this layer.
* Which parameters were tuned? How were they adjusted and why? Batch size and learning rate were tuned but only the learning rate resulted in a significant improvement on model performance. After training the model a few times with a learning rate of 0.01, I realized that it was overfitting too fast, and the validation loss was varying very quickly during few epochs, so I decided to lower the learning rate until I had more stable training, with the loss and accuracy values in the validation set varying less. By doing this, I got a jump from 89% accuracy to 95%. I also tried to increase and decrease the batch size (to 64 and 16), but I didn't got any improvement.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Since we are working with images, convolution layers that learns filters to extract features like horizontal and vertical lines and then stacking them to make abstract shapes, is a good choose. A dropout layer can help to create a more robust model, because with the dropout the network tends to depend less on specific filters / neurons of the network (since they can be "turned off" during training).
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first image was the only one that the network did not classify correctly, it might be difficult to classify because the dataset does not have too much examples of that class, so, it is not robust enough to distinguish between the Speed limit (100km/h) sign and Dangerous curve to the right sign, which are similar.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			         |     Prediction	        					| 
|:----------------------:|:--------------------------------------------:| 
| Yield      		     | Yield  									    | 
| Wild animals crossing  | Wild animals crossing 					    |
| Turn left ahead		 | Turn left ahead								|
| Speed limit (100km/h)	 | Dangerous curve to the right					|
| No entry			     | No entry      							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.36%. To have a more realistic result, getting closer to the accuracy in the test set, it would be necessary to test more images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

In 4 out of 5 images, the model is very sure of the correct class, in the only image in which it is not very sure about the class of the image, Speed limit plate (100km/h) with a probability of 60%, the prediction is wrong. 

### **Image 1 (Yield)**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| Yield    									    | 
| 0.00     				| Speed limit (20km/h) 						    |
| 0.00					| Speed limit (30km/h)						    |
| 0.00	      			| Speed limit (50km/h)					 		|
| 0.00				    | Speed limit (60km/h)      					|

### **Image 2 (Wild animals crossing)**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| Wild animals crossing    						| 
| 0.00     				| Right-of-way at the next intersection 		|
| 0.00					| Double curve						            |
| 0.00	      			| Speed limit (80km/h)					 		|
| 0.00				    | Speed limit (60km/h)      					|

### **Image 3 (Turn left ahead)**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| Turn left ahead    						    | 
| 0.00     				| Ahead only 		                            |
| 0.00					| Keep right						            |
| 0.00	      			| Go straight or right					 		|
| 0.00				    | General caution      					        |

### **Image 4 (Speed limit (100km/h))**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60.96         		| Dangerous curve to the right    				| 
| 20.69     			| End of no passing 		                    |
| 18.33					| Slippery road						            |
| 0.00	      			| Speed limit (80km/h)					 		|
| 0.00				    | ehicles over 3.5 metric tons prohibited       |

### **Image 5 (No entry)**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| No entry    						            | 
| 0.00     				| No passing 		                            |
| 0.00					| No vehicles						            |
| 0.00	      			| Speed limit (20km/h)					 		|
| 0.00				    | Speed limit (60km/h)      					|

