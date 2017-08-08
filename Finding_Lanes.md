

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from collections import deque
import glob
import pickle

%matplotlib inline
```

## Camera Calibration


```python
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera_cal/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        plt.imshow(img)
        
```


![png](output_2_0.png)



```python
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1][1:], None,None)
pickle.dump({'mtx': mtx, 'dist': dist}, open("cam_calibration.p", "wb"))
dst = cv2.undistort(img, mtx, dist, None, mtx)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
fig.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
print('...')
```

    ...
    


![png](output_3_1.png)



```python
plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.title("Undistorted Image")
img = cv2.imread('test_images/test1.jpg')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(1, 2, 2)
undist_img = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
plt.title("Undistorted Image")
```




    <matplotlib.text.Text at 0x7f393416f5f8>




![png](output_4_1.png)


## Create thresholded binary images


```python
def gaussian_blur(img, sigma):
    kernel_size = 11
    return cv2.GaussianBlur(img, (kernerl_size, kernel_size), sigma)

def abs_sobel_thresh(img, thresh_min=25, thresh_max=255, sobel_kernel=7, orient='x'):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    bin_output = np.zeros_like(scaled_sobel)
    bin_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return bin_output

def mag_thresh(img, sobel_kernel=7, thresh_min=25, thresh_max=255):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(magnitude)/255 
    magnitude = (magnitude/scale_factor).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    bin_output = np.zeros_like(magnitude)
    bin_output[(magnitude >= thresh_min) & (magnitude <= thresh_max)] = 1

    return bin_output

def directional_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    gradient = np.arctan2(np.abs(sobely), np.abs(sobelx))
  
    dir_binary = np.zeros_like(gradient)
    dir_binary[(gradient > thresh_min) & (gradient <= thresh_max)] = 1
                       
    return dir_binary

def color_thresh(img, thresh_min=0, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    s = hls[:, :, 2]
    bin_output = np.zeros_like(s)
    bin_output[(s > thresh_min) & (s <= thresh_max)] = 1
    return bin_output

def transform_perspective(img, src, dst):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, M_inv
```


```python
src = np.float32([[490, 482],[810, 482],
                 [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                  [1250, 720],[40, 720]])
apex, apey = 360, 254
offset_far = 60
offset_near = 10
src = np.float32([[int(apex-offset_far),apey],
                  [int(apex+offset_far),apey],
                  [int(0+offset_near),390],
                  [int(720-offset_near),390]])
dst = np.float32([[0,0],[720,0],[0,405],[720,405]])

#def pipeline(img):
#    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
#    warped = transform_perspective
```


```python
params = pickle.load( open( "cam_calibration.p", "rb" ) )
mtx = params['mtx']
dist = params['dist']
for fname in glob.glob("test_images/test*.jpg"):
    #fname = os.path.join('test_images',fname)
    img = mpimg.imread(fname)
    img = cv2.resize(img, (720, 405))
    
    warped, M, M_inv = transform_perspective(img, src, dst)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(warped)
    ax2.set_title('Perspective transformed', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

```


![png](output_8_0.png)



![png](output_8_1.png)



![png](output_8_2.png)



![png](output_8_3.png)



![png](output_8_4.png)



![png](output_8_5.png)



```python

for fname in glob.glob("test_images/test*.jpg"):
    image = mpimg.imread(fname)
    image = cv2.resize(image, (720, 405))
    warped, M, M_inv = transform_perspective(image, src, dst)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    grad_x = abs_sobel_thresh(warped, orient='x', sobel_kernel=3, thresh_min=10, thresh_max=230)
    grad_y = abs_sobel_thresh(warped, orient='y', sobel_kernel=3, thresh_min=10, thresh_max=230)
    mag_binary = mag_thresh(warped, sobel_kernel=3, thresh_min=30, thresh_max=150)
    dir_binary = directional_thresh(warped, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
    hls_binary = color_thresh(warped, thresh_min=85, thresh_max=255)
    combined = np.zeros_like(dir_binary)
    
    combined[((grad_x == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    f.tight_layout()
    ax1.imshow(warped)
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(combined,cmap='gray' )
    ax2.set_title('Thresholded', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)



```python
def apply_thresholds(image):
    image = cv2.resize(image, (720, 405))
    warped, M, M_inv = transform_perspective(image, src, dst)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    grad_x = abs_sobel_thresh(warped, orient='x', sobel_kernel=3, thresh_min=10, thresh_max=230)
    grad_y = abs_sobel_thresh(warped, orient='y', sobel_kernel=3, thresh_min=10, thresh_max=230)
    mag_binary = mag_thresh(warped, sobel_kernel=3, thresh_min=30, thresh_max=150)
    dir_binary = directional_thresh(warped, sobel_kernel=3, thresh_min=0.7, thresh_max=1.3)
    hls_binary = color_thresh(warped, thresh_min=85, thresh_max=255)
    combined = np.zeros_like(dir_binary)
    
    combined[((grad_x == 1) & (hls_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined, M_inv
```


```python
histogram = np.sum(combined[int(combined.shape[0]/2):,:], axis=0)

# Peak in the first half indicates the likely position of the left lane
half_width = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:half_width])

# Peak in the second half indicates the likely position of the right lane
rightx_base = np.argmax(histogram[half_width:]) + half_width

print(leftx_base, rightx_base)
plt.plot(histogram)
```

    88 612
    




    [<matplotlib.lines.Line2D at 0x7f391f6938d0>]




![png](output_11_2.png)



```python
def find_window_centroids(binary_warped, nwindows=9, margin=100):
    """Implementation provided by Udacity.
    """

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint]) + 30
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, ploty, left_fitx, right_fitx

```


```python
def search_around_previous_fit(binary_warped, left_fit, right_fit):
    #From udacity
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, ploty, left_fitx, right_fitx

```


```python
class Lane(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        self.recent_fits = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # x values in windows
        self.windows = np.ones((3, 12)) * -1

```


```python
class LaneFinder(object):
    """Class to find left and right lanes.
    """
    def __init__(self):
        self.left_lane = Lane()
        self.right_lane = Lane()

    @staticmethod
    def get_curvature(y, fitx):
        y_eval = np.max(y)
        # Define conversions in x and y from pixels space to meters
        # assume the lane is about 30 meters long and 3.7 meters wide
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(y * ym_per_pix, fitx * xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad

    @staticmethod
    def get_position(pts_left, pts_right, image_shape=(405, 720)):

        #  Find the position of the car from the center
        # It will show if the car is 'x' meters from the left or right
        left_mean = np.mean(pts_left)
        right_mean = np.mean(pts_right)
       
        xm_per_pix = 3.7/700
        camera_pos = (image_shape[1]/2)-np.mean([left_mean, right_mean])
        return camera_pos*xm_per_pix

    @staticmethod
    def sanity_check(lane, curverad, fitx, fit):
        lane.current_fit = fit

        if abs(curverad - 2000) / 2000 < 2:
            lane.detected = True

            #Keep a running average over 3 frames
            if len(lane.recent_xfitted) > 5 and lane.recent_xfitted:
                lane.recent_xfitted.pop()
                lane.recent_fits.pop()

            lane.recent_xfitted.append(fitx.reshape(1,-1))
            lane.recent_fits.append(fit.reshape(1,-1))

            if len(lane.recent_xfitted) > 1:
                lane.bestx = np.mean(np.vstack(lane.recent_xfitted),axis=1)
                lane.best_fit = np.mean(np.vstack(lane.recent_fits),axis=1)

            lane.bestx = fitx
            lane.best_fit = fit

            return lane.bestx

        else:
            lane.detected=False

        return  lane.bestx if lane.bestx is not None else lane.current_fit

    # Takes care of fitting right and left lanes to the image.
    def _fit_lanes(self, image):

        left_lane, right_lane = self.left_lane, self.right_lane

        if left_lane.detected and right_lane.detected:
            left_fit, right_fit, yvals, left_fitx, right_fitx = \
                search_around_previous_fit(image, left_lane.best_fit, right_lane.best_fit)
        else:
            left_fit, right_fit, yvals, left_fitx, right_fitx = find_window_centroids(image)

        # Find curvatures
        left_curverad = self.get_curvature(yvals, left_fitx)
        right_curverad = self.get_curvature(yvals, right_fitx)

        left_fitx = self.sanity_check(left_lane, left_curverad, left_fitx, left_fit)
        right_fitx = self.sanity_check(right_lane, right_curverad, right_fitx, right_fit)

        return yvals, left_fitx, right_fitx, left_curverad

    @staticmethod
    def draw_poly(image, warped, yvals, left_fitx, right_fitx, Minv, curvature):
        image = cv2.resize(image, (720, 405))
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))]) + 30
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        # Put text on an image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Radius of Curvature: {} m".format(int(curvature))
        cv2.putText(result,text,(10,30), font, 1,(255,255,255),2)
        # Find the position of the car
        pts = np.argwhere(newwarp[:,:,1])
        position = LaneFinder.get_position(pts_left, pts_right)
        if position < 0:
            text = "Vehicle is {:.2f} m left of center".format(-position)
        else:
            text = "Vehicle is {:.2f} m right of center".format(position)
        cv2.putText(result,text,(10,60), font, 1,(255,255,255),2)
        return result
    
    def process_image(self, image):
        #image = cv2.resize(image, (720, 405))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bin_img, M_inv = apply_thresholds(image)
        y_vals, lt_fit, rt_fit, curvature = self._fit_lanes(bin_img)
        return self.draw_poly(image, bin_img, y_vals, lt_fit, rt_fit, M_inv, curvature)
```


```python
lf = LaneFinder()
test_img = cv2.imread('./test_images/test6.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
img_out = lf.process_image(test_img)
plt.imshow(img_out)
```




    <matplotlib.image.AxesImage at 0x7f392444e470>




![png](output_16_1.png)



```python
LF = LaneFinder()
write_output = 'project_video_out.mp4'
clip = VideoFileClip("project_video.mp4")

write_clip = clip.fl_image(LF.process_image)#.subclip(20, 23)
write_clip.write_videofile(write_output, audio=False, progress_bar=False)

```

    [MoviePy] >>>> Building video project_video_out.mp4
    [MoviePy] Writing video project_video_out.mp4
    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_out.mp4 
    
    
