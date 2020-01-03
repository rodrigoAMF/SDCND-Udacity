
# **Finding Lane Lines on the Road** 

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied a Gaussian Blur with kernel size of 5 in order to reduce the noise from the image. After that, I used a Canny Edge Detector with low threshold and high threshold of 20 and 90 respectively to detect the edges of the image. Since the lane lines usually stays at the same position on the image, a mask were applied to black out some parts of the image, reducing the number of edges of the image and consequently the number of potential lane lines. Finally to detect the lane lines, I applied a Hough Transform, on the masked image, to find the lines.    

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first detecting the lines belonging to each lane, it was done by calculating the slope of each line, if it has a slope less than or equal to -0.5 it belongs to the left lane, else if the slope is greater than or equal to 0.5 it belongs to the right lane, any other lines that doesn't met these conditions were discarded. With the lines of each lane, for each lane, I used the point with the lowerest Y value as one of the points of the final line. Using that point and the point with the highest Y, I calculated the coefficients of the line and used the equation of the line to find the X value for a Y equals to the height of the image, I did this because sometimes we can have a not continuous lane that can be slightly above the bottom of the image and I wanted that the line always start at the bottom of the image. 

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming could be what would happen when we find sharp turns, since the pipeline draw one single line, in that case, it will not detect good lane lines.

Another shortcoming could be when the car finds a work zone and need to change lanes with only a traffic sign, if the car relies only on the lane detection as it was done in the pipeline, it will not be able to do so.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use the values of the points of the lines, previously detected, to make an average with the new values in order to avoid abrupt changes in lane detection

Another potential improvement could be create a visual tool to test hyperparameters for the hough transform in real time, making easier to do tests, and improving the lane line detection.
