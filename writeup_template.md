
# Advanced Lane Finding Project

The goal of this project is to provide a pipeline to detect lanes in a video captured while driving a car.
These were the steps involved in the pipeline:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report_images/output_3_1.png "Undistorted"
[image2]: ./report_images/output_4_1.png "Road Transformed"
[image3]: ./report_images/output_8_3.png "Birds-eye"
[image4]: ./report_images/output_9_3.png "Warp Example"
[image5]: ./report_images/output_12_2.png "Histogram"
[image6]: ./report_images/poly_fit.png "Polynomial Fit"
[image7]: ./report_images/output_16_1.png "Polynomial Fit"
[video1]: ./project_video_out.mp4 "Video"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration can be found in code cell 2-3 of Jupyter notebook located at "./Finding_Lanes.ipynb". The images used for calibration are present in folder called `camera_cal`.

Using the chessboard images in `camera_cal`, Calibration matrix and distortion coefficients are calculated. These values are saved to disk for future use. The two main functions used are `cv2.findChessboardCorners` and `cv2.drawChessboardCorners`.


#### 1. Provide an example of a distortion-corrected image.

The images below show the effect of distortion-correction using the calibration matrix and distortion coefficients.
![alt text][image1]

Similarly, the images below show the effect of the distortion-correction on the road image.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The first step of the pipeline is to produce a binary image which highlights lanes. To achieve this, I used directional Sobel thresholds, Gradient Magnitude Threshold and S channels thresholds. The parameters for these thresholds were manually tuned. Before applying all these thresholds, perspective transforms was performed to convert the image to "birds-eye" view.

![alt text][image3]
This is the result of binary thresholding.
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in cell 5 of jupyter notebook `Finding_Lanes.ipynb`  The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[ 300.  254.]
                  [ 420.  254.]
                  [  10.  390.]
                  [ 710.  390.]])
dst = np.float32([[0,0],[720,0],[0,405],[720,405]])
```

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `find_window_centroids` (cell 11) and `search_around_previous_fit` (cell 12) were used to identify lane lines and fit a second order polynomial to the lanes. The bottom of the left and right lanes are found using histogram of binary image. The two peaks in the histogram is identified and second order polynomial are fitted to these positions. In order to make the lane detection more robust, the last 5 fits for the lanes are cached and averaged.

Below is an example of histogram.
![alt text][image5]

An example of polynomial fit:
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and car positioning was calculated using `get_curvature` and `get_position` functions (cell 14). The radius of curvature is base on this site: `http://www.intmath.com/applications-differentiation/8-radius-curvature.php`. The parameters to convert from pixels to meters were based on the data provided by Udacity and is based upon the assumption that a lane is about 30 m long and 3.7m wide.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`draw_poly` function in cell 14 plots the detected lanes and curvature angle and vehicle position on the image.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of this project was finding the parameters for thresholds. Also, the pipeline doesn't seem to work for harder videos due to sharper turns. One way this could be improved is by taking smaller section to take the transform. I believe snow on the road would also be an issue. The segmentation procedure can be improved to make it robust in different lightning conditions. Perhaps a combination of Machine learning techniques would help in lane segmentation.  
