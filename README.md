## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration16.jpg "Example of distortion correction: original image (left) and corrected image (right)"
[image2]: ./output_images/undistorted_test5.jpg "Example of distortion correction: original image (left) and corrected image (right)"
[image3]: ./output_images/birdseyeView_test5.jpg "Example of perspective transformation: undistorted image car-view(left) and undistorted image birds-eye view (right)"
[image4]: ./output_images/image_pipeline_example.png "Binarization"
[image5]: ./output_images/lane_finding_example.png "Convolution search for detecting lane center and width"
[image6]: ./output_images/test5_debug.jpg "Lane detection. Different colors represent the three binary channels. Blue dots are the detected lane centers."
[image7]: ./output_images/test5_processed.jpg "Result of convolution search"
[image8]: ./output_images/lookahead_example.jpg "Look-ahead lane detection. Different colors represent the three binary channels. Blue dots are the detected lane centers."


[video1]: ./output_images/project_video_processed.mp4 "Video"
[video2]: ./output_images/challenge_video_processed.mp4 "Video"
[video3]: ./output_images/harder_challenge_video_processed.mp4 "Video"


###### Here I will describe how I addressed each of the [goals](https://review.udacity.com/#!/rubrics/571/view) for this project
---

### 1. Camera Calibration and perspective transformation

To perform the camera calibration, distortion correction, and perspective transformations between the "car's view" and the "bird's eye view" (bev), I created a Camera class (located in `camera.py`).
This class handles loading camera calibration images, computing the camera matrix and distortion coefficients, and computing the perspective transformation matrices given source (car-view) and destination (bev) points.

Below is an example of an undistorted camera calibration image, and undistorted test image, and an image transformed into birds-eye view.


![Example of distortion correction: original image (left) and corrected image (right)][image1]

![Example of distortion correction: original image (left) and corrected image (right)][image2]

![Example of perspective transformation: undistorted image car-view(left) and undistorted image birds-eye view (right)][image3]

### 2. Image binarization

Image binarization is handled by the class RoadBinarizer located in `roadBinarizer.py`. The pipeline I used is graphically described below.

I start with converting the image to Lab color and creating three color channels: L, b, and (L+b)/2. The L channel is then converted to the birds-eye view (bev) and thresholded. I then take the x-derivative of the b and (L+b)/2 channels, convert them into bev, and then apply a filter that only lets light on dark lines pass through. (This filter used the fact that the bright lanes will have a positive x-derivative on the left and a negative derivative on the right. Credit for this idea goes to [balancap](https://github.com/balancap/SDC-Advanced-Lane-Finding).) These three channels are then combined into a single channel by adding them together and then squaring the results, which will give more weight to regions where all three color channels overlap when searching for the lines.

![Binarization][image4]

### 3. Lane identification
In this project I search-for/detect the center and width of the lane, instead of directly searching for the left and right lane lines. The primary search is done by convolution filters and a faster look-ahead method is performed when the binarized image has low density and the approximate location of the lane is known.

The code for this is all contained in two classes named `Road` and `Line` (located in `road.py`). The Line class contains the history of the points used to describe the lane center and width as well as the methods for fitting these points with polynomials, and determining the lane curvature and position. The `Road` class contains instances of each class `Camera`, `RoadBinarizer`, and `Line`, and performs the search and logic for detecting the road lines, and drawing annotated images.

#### A. Lane detection by convolution
I detect the center and width of the lane using a convolution search as graphically described below. I first split up the image into several horizontal slices and for each slices I convolute a filter that has two positive regions separated by a large negative region; the positive regions are approximately the width of the road lines (50 pixels) and distance between the center of the positive regions (gap) represents the width of the lane. By convolving the filter using several different gap sizes, I can find the center of the lane and the width of the lane by looking for the convolution with the maximum value. *(The code for the method is in `_initialize_line_centroids` in class `Road`.)*

![Convolution method for detecting lane center and width][image5]

This method is more robust in detecting the lane in the presence of noise as it computes the best locations for the left and right lines simultaneously.

After getting the center and width of the lane for each slice of the image. I fit the center points with a second order polynomial and I fit the width of the lane using a first order polynomial; both of these use a robust optimization method ('soft_l1' using least_squares from the scipy library). An example of the road detected with this method can be seen below (both in birds-eye vies and the cars-view); the blue points are the center points and the yellow points are the center points +- half the width.

![Convolution method for detecting lane center and width][image6]
![Convolution method for detecting lane center and width][image7]

#### B. Look-ahead detection

After the lane position is known from using the convolution method above, a faster search is used (when the density of the binary image is low). This method using the center-of-mass of many small windows around the expected location of the left and right lane to compute the lane center and width. The small windows can be seen as the boxes in the below image, and the center of mass of the windows are shown as the yellow dots. These center of mass points are then used to compute new lane center and width points for updating the fit.

![Look-ahead method for detecting lane center and width][image8]

#### C. Short distance Look-ahead
When the detected points of the lane do not extend to the top of the image, using look-ahead detection all the way to the top can lead to bad detection of the lane. For this reason, look-ahead detection is only performed a short distance past the highest detected points.

#### D. History
The points describing the lane and the coefficients describing the lane center and width are each stored for several iterations. The mean of them are used for the lane of the current frame. (This is all implemented in the `Lane` class.)

---

### Results

The processed videos can be seen at the following links: [project video](./project_video_processed.mp4), [challenge video](./challenge_video_processed.mp4), and [harder challenge video](./harder_challenge_video_processed.mp4). In addition, there are "debug" versions of the processed videos where the binarized image in birds-eye view is shown with the projected lane location: [project video debug](./project_video_debug.mp4), [challenge video debug](./challenge_video_debug.mp4), and [harder challenge video debug](./harder_challenge_video_debug.mp4).  

---

### Discussion

Looking at the [debug version of the harder challenge video](./harder_challenge_video_debug.mp4) I think that my method is extracting most of the lane information possible from the binarized image; however, I did not program the method to work on the U-turn where only one lane is visible.

What I have implemented may be a good starting point, but it needs smarter logic. In particular, it needs to actually keep track of the left and right lane, and even lane lines for the lanes next to the one the car is in. This would allow the tracking to work more robustly, and in particular when only one line of the current lane is visible. Additionally, the processing is pretty slow, so it would need to be sped up for actual use.
