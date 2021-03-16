import os.path
import pickle

import numpy as np
import cv2

import matplotlib.pyplot as plt

def undistortImage(image, cameraData):
    return cv2.undistort(image, cameraData['mtx'], cameraData['dist'], None, cameraData['mtx'])

def doPerspectiveTransformation(image, cameraData):
    img_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, cameraData['M'], img_size, flags=cv2.INTER_LINEAR)

def undoPerspectiveTransformation(image, cameraData):
    img_size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, cameraData['Minv'], img_size, flags=cv2.INTER_LINEAR)

def checkIfPickleHasEverything():
    with open('camera.pickle', 'rb') as handle:
        cameraPickle = pickle.load(handle)
    if all (data in cameraPickle for data in ("mtx","dist","rvecs","tvecs","M","Minv")):
        return True
    return False

def loadCameraData():
    if not os.path.isfile('camera.pickle'):
        raise Exception("camera.pickle does not exist!!")
    with open('camera.pickle', 'rb') as handle:
        cameraData = pickle.load(handle)
    return cameraData

def saveCameraData(newCameraData):
    # If there is a cameraData.picke file, just overwrite part of the atributes
    if os.path.isfile('camera.pickle'):
        cameraData = loadCameraData()
        for key, value in newCameraData.items():
            cameraData[key] = value
    else:
        cameraData = newCameraData
    # Save the cameraData to camera.pickle
    with open('camera.pickle', 'wb') as handle:
        pickle.dump(cameraData, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
# Edge detection
def absolute_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Expect a grayscale (1 channel image)
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.abs(sobel)
    # To normalize the values and they stay in the range of 0-255
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def magnitude_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Expect a grayscale (1 channel image)
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    scaled = np.uint8(255*mag/np.max(mag))
    # Apply threshold
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return mag_binary

def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Expect a grayscale (1 channel image)
    # Calculate gradient direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # Get absolute values
    absSobelx = np.abs(sobelx)
    absSobely = np.abs(sobely)
    
    dirGrad = np.arctan2(absSobely, absSobelx)
    
    # Apply threshold
    dir_binary = np.zeros_like(dirGrad)
    dir_binary[(dirGrad >= thresh[0]) & (dirGrad <= thresh[1])] = 1
    return dir_binary

def convert_to_HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def get_channel(img, channel='R'):
    # image can be on RGB or HLS color channel
    if channel == 'R' or channel == 'H':
        return img[:,:,0]
    elif channel == 'G' or channel == 'L':
        return img[:,:,1]
    elif channel == 'B' or channel == 'S':
        return img[:,:,2]
    
def color_thresh(img, thresh=(0,255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def plot_side_by_side(img1, img2, title1="Image 1", title2="Image 2", gray=False):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax.set_title(title1)
    if not gray:
        ax.imshow(img1)
    else:
        ax.imshow(img1, cmap='gray')
    ax2.set_title(title2)
    if not gray:
        ax2.imshow(img2)
    else:
        ax2.imshow(img2, cmap='gray')
    plt.show()

def find_edges(img, cameraData):
    undistorted = undistortImage(img, cameraData)
    # Detect edges with the color channels
    s_channel = get_channel(convert_to_HLS(undistorted), 'S')
    r_channel = get_channel(undistorted, 'R')
    
    binary_S_channel = color_thresh(s_channel, [170, 255])
    binary_R_channel = color_thresh(r_channel, [226, 255])
    
    combined_binary_color_image = np.zeros_like(binary_S_channel)
    combined_binary_color_image[(binary_S_channel == 1) | (binary_R_channel == 1)] = 1
    
    # Detect edges with the sobel operator
    binary_sobelX = absolute_sobel_thresh(s_channel, 'x', 3, [20, 255])
    binary_sobelY = absolute_sobel_thresh(s_channel, 'y', 3, [40, 100])
    binary_mag = magnitude_thresh(s_channel, 3, [40, 100])
    binary_dir = direction_threshold(s_channel, 3, [0.65, 1.2])
    
    combined_binary_sobel_image = np.zeros_like(binary_sobelX)
    combined_binary_sobel_image[((binary_sobelX == 1) & (binary_sobelY == 1)) | ((binary_mag == 1) & (binary_dir == 1))] = 1
    
    edges = np.zeros_like(combined_binary_sobel_image)
    edges[(combined_binary_color_image == 1) | (combined_binary_sobel_image == 1)] = 1
    
    return edges

def find_edges_warped(img, cameraData):
    undistorted = undistortImage(img, cameraData)
    edges = find_edges(undistorted, cameraData)
    
    return doPerspectiveTransformation(edges, cameraData)
    

def plot_image(img, title=None, gray=False):
    plt.figure(figsize = (15,15))
    if title is not None:
        plt.title(title)
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    
def hist(img):
    img = img/255.0
    # img.shape = (y, x) 720, 1280
    bottom_half = img[img.shape[0]//2:, :]
    
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

def measure_curvature_pixels(left_coefficients, right_coefficients, y_values):
    y_eval = np.max(y_values)
    
    # Calculation of R_curve (radius of curvature)
    left_curverature = ((1 + (2*left_coefficients[0]*y_eval + left_coefficients[1])**2)**1.5) / np.absolute(2*left_coefficients[0])
    right_curverature = ((1 + (2*right_coefficients[0]*y_eval + right_coefficients[1])**2)**1.5) / np.absolute(2*right_coefficients[0])
    
    return left_curverature, right_curverature

def fit_polynomial(left_x, left_y, right_x, right_y, img_shape):    
    # coefficients from Ex: 2x² + 5x + 4 = [2, 5, 4]
    # np.polyfit finds the values that solves a n degree equation that describes the given points
    left_coefficients = np.polyfit(left_y, left_x, 2)
    right_coefficients = np.polyfit(right_y, right_x, 2)
    
    # generate y values to plot the lane line to later get x position and plot the line
    # Ex: f(y) = 2y² + 5y + 4. I'm here generating values for y
    # This function generate img_shape.shape[0](720) values from [0, img_shape.shape[0]-1(719)]
    y_values = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # get the x values from the y values previsiously generated
    left_x_values  = left_coefficients[0] * y_values**2 + left_coefficients[1] * y_values + left_coefficients[2]
    right_x_values = right_coefficients[0] * y_values**2 + right_coefficients[1] * y_values + right_coefficients[2]
    
    return left_x_values, right_x_values, y_values