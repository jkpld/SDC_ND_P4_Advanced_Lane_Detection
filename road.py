import numpy as np
import cv2

from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import least_squares

from collections import deque

from camera import Camera
from roadBinarizer import RoadBinarizer

from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc():
    return (time() - _tstart_stack.pop())

def poly2res(x, t, y):
    return x[0]*t**2 + x[1]*t + x[2] - y

def poly1res(x, t, y):
    return x[0]*t + x[1] - y

# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self, history_size=5, history_sigma=3):
        # use look-forward searching
        self.use_lookahead = False
        # was the line detected in the last iteration?
        self.detected = False

        # Set up history for saving the control points from the last
        # `history_size` observations. history_sigma is the sigma value
        # used to create the gaussian weights applied to the control points
        # when fitting a polynomial (see update() below for example)
        self.history_sigma = history_sigma
        self.control_points = deque(maxlen=history_size)

        # variable to store the lane polynomial coefficients
        self.line_coeff = deque(maxlen=history_size)
        self.line_width_coeff = deque(maxlen=history_size)
        # polynomial function usine the line_coeff values
        self.line_fn = None
        self.line_width_fn = None

        self.robustFitting = False # If using robust fitting, then hisotry_sigma is not used
        self.lastErrors = (None, None)

        self.default_lane_width = 750

    # Radius of curvature of the line in some units
    def radius_of_curvature(self, y, ys=1, xs=1):
        # ys is a y scale factor (meters per pixel in y direction)
        # x2 is a x scale factor (meters per pixel in x direction)
        if len(self.line_coeff) is 0:
            return 0.0

        A = self.line_fn.c[0]*xs/ys**2
        B = self.line_fn.c[1]*xs/ys
        y = y*ys
        R = (1 + (2*A*y + B)**2 )**1.5/np.abs(2*A)
        return R

    # Base location of the line in some units
    def line_base_pos(self, y, xs=1):
        if self.line_fn is None:
            return None

        x_pos = self.line_fn(y) * xs
        # Note this position is relative to the image origen
        return x_pos

    # Create property for history_size that will return the max length of
    # the control_points deque
    @property
    def history_size(self):
        return self.control_points.maxlen

    # Add a setter method for the history_size which will reinitialize a
    # new control_points deque with a max length of history_size using the
    # old control_points deque
    @history_size.setter
    def history_size(self, new_size):
        self.control_points = deque(self.control_points, maxlen=new_size[0])
        self.line_coeff = deque(self.line_coeff, maxlen=new_size[1])
        self.line_width_coeff = deque(self.line_width_coeff, maxlen=new_size[2])

    # Compute the control point weights. The weights will fall off as a
    # gaussian (with sigma=history_sigma) as the control points get older
    def control_point_weights(self):
        # The number of control points from each observation
        cp_sizes = [cp.shape[0] for cp in self.control_points]
        # The weight value for each observation
        weights = np.exp(-np.flipud(np.arange(0,len(cp_sizes)))**2 / (2 * self.history_sigma**2.))
        # Replicate the weights and return
        return np.repeat(weights, cp_sizes)

    def check_values(self, line_coeff, line_width_coeff, thresholds=(5,5)):
        if self.line_fn is not None:
            err_lane = np.mean(np.abs((self.line_fn.c - line_coeff)/line_coeff))
            err_width = np.mean(np.abs((self.line_width_fn.c - line_width_coeff)/line_width_coeff))

            goodValues = True if err_lane < thresholds[0] and err_width < thresholds[1] else False
            self.lastErrors = (err_lane, err_width)
        else:
            goodValues = True
            self.lastErrors = (0.0,0.0)
        return goodValues

    # Update the lane position
    def _update(self):
        if self.detected:
            # reset last detection counter
            self.last_detection = 0
            # Combine all of the control points into one array
            cp = np.concatenate(self.control_points, axis=0)

            if self.robustFitting:
                # Robust fitting does not use weights, so if there are
                # points we are very confident about, then we will
                # replicate the points
                cp = np.repeat(cp, cp[:,3].astype(np.int32),axis=0)

                # Fit the center position and the width

                linewidth = least_squares(poly1res, np.array((0,np.mean(cp[:,2])/2)), loss='soft_l1', f_scale=10, args=(cp[:,1],cp[:,2]/2))
                line_width_coeff = linewidth.x

                # If the range of the y control-points is less than 150,
                # then there is likely not enough information to acurately
                # get lane curvature or lane width, all we can hope for
                # is the linear lane center
                if np.ptp(cp[:,1]) < 150:
                    linecenter = least_squares(poly1res, np.array((0,0,np.mean(cp[:,0]))), loss='soft_l1', f_scale=10, args=(cp[:,1],cp[:,0]))
                    linecenter = linecenter.x
                    line_coeff  = np.array([1e-10, linecenter[0],linecenter[1]])
                    linewidth[0] = 0
                    linewidth[1] = self.default_lane_width/2
                else:
                    linecenter = least_squares(poly2res, np.array((0,0,np.mean(cp[:,0]))), loss='soft_l1', f_scale=10, args=(cp[:,1],cp[:,0]))
                    line_coeff = linecenter.x


            else:
                # Fit control points with 2nd order polynomial using the
                # weights returned by control_point_weights()
                w = self.control_point_weights()
                line_coeff = np.polyfit(cp[:,1], cp[:,0], 2, w=w)
                line_width_coeff = np.polyfit(cp[:,1], cp[:,2]/2, 1, w=w)

            goodValues = self.check_values(line_coeff, line_width_coeff)

            # if goodValues:
            # Add the coefficients to the history
            self.line_coeff.append(line_coeff[None,:])
            self.line_width_coeff.append(line_width_coeff[None,:])

            # Using the mean value of the coefficient history
            # for the center/width functions
            lc = np.concatenate(self.line_coeff, axis=0)
            lwc = np.concatenate(self.line_width_coeff, axis=0)

            if len(self.line_coeff) > 1:
                lc = np.mean(lc, axis=0)
                lwc = np.mean(lwc, axis=0)

            # Use the coefficients to create a function (simply a convenience)
            self.line_fn = np.poly1d(lc.flatten())
            self.line_width_fn = np.poly1d(lwc.flatten())

            # Set the look forward flag to true
            self.use_lookahead = True
        else:
            # remove the oldest control points from the history
            num_control_points = len(self.control_points)
            if num_control_points > 0:
                self.control_points.popleft()
            # If there are no historical control points, then reinitialize
            # the lane
            if num_control_points < 1:
                self.use_lookahead = False

    def addObservation(self, control_points):
        # If there are at least two control points, then the
        # observation could be good
        if len(control_points) > 1:
            cp = np.array(control_points,ndmin=2) # Convert to numpy array

            self.control_points.append(cp)
            self.detected = True
        else:
            # If there were less than two control points found, then
            # the observation failed
            self.detected = False
        # Update the line
        self._update()

    def reset(self):
        self.detected = False
        self.use_lookahead = False
        self.control_points.clear()
        self.line_coeff.clear()
        self.line_width_coeff.clear()
        self.line_fn = None
        self.line_width_fn = None
        self.lastErrors = (None, None)


class Road():

    sliding_window_width = 50
    sliding_window_height = 25
    search_margin = 100
    minimum_density = 0.01

    def __init__(self):
        self.CarCam = Camera()
        self.Binarizer = RoadBinarizer()
        self.lane_center = Line(history_size=2, history_sigma=2)

        self.ym_per_pix = 3/100 # in birds-eye view
        self.xm_per_pix = 3.7/750 # in birds-eye view
        self._is_first_image = True
        self.debug = False

    def reset(self):
        self.lane_center.reset()
        # self._is_first_image = True

    @staticmethod
    def find_laneCenterWidth(im, w, search_gaps, bestConvOutput=[]):
        # find the lane center and width by convoluting with a filter
        # the filter is something like:
        #
        # -----|          gap           |-------
        # .....|........................|.......   0
        #      |------------------------|
        #
        # where the gap changes size. Using this we can find the center
        # position of the lane as well as the width of the lane.
        # This works quite robustly, but it requires both lanes (at least
        # using the 'valid' mode on the convolution.)
        # Note, this method could be used even when there is only one lane
        # present, but must use the full convolution and then search for
        # one to three peaks depending on the number of lanes present--and
        # then code some logic to sort the peaks out.
        window = np.ones(w)/(2*w)

        lcenter = None
        lwidth = None
        maxConv = -10000
        for gap in search_gaps:
            convFilter = np.hstack((window, -np.ones(int(gap - w))/(gap-w), window))
            conv = np.convolve(convFilter, im, mode='valid')

            if np.max(conv) > maxConv:
                lwidth = gap
                lcenter = np.argmax(conv) + convFilter.size/2
                maxConv = np.max(conv)
                bestConvOutput = conv

        return lcenter, lwidth, maxConv

    def _initialize_line_centroids(self, image, startAt=(0,None,None,None)):

        lane_centers = [] # Store the lane information
        imgH, imgW = image.shape[0:2]

        # search range of gaps
        search_gaps = np.arange(300,855,25)

        if startAt[1] is None:
            use_lookahead = self.lane_center.use_lookahead
        else:
            use_lookahead = False

        # Determine approximate center and width of lane to start
        num_slices = (int)(imgH/self.sliding_window_height)
        if startAt[1] is None:
            if use_lookahead:
                # If we are looking ahead, then use the previously
                # computed lane to find the expected centeres and widths
                slice_centers = imgH - (np.arange(num_slices)+0.5)*self.sliding_window_height
                expected_centers = self.lane_center.line_fn(slice_centers)
                expected_widths = 2*self.lane_center.line_width_fn(slice_centers)

                # Look-ahead a maximum of three steps
                if len(self.lane_center.control_points) > 0:
                    cp = np.concatenate(self.lane_center.control_points, axis=0)
                    minY = np.min(cp[:,1]) - 3*self.sliding_window_height
                else:
                    # This condition should not occur because if there are
                    # no control points, then use_lookahead should be False
                    minY = 0
            else:
                image_layer = np.mean(image[int(3*imgH/5):,:], axis=0)
                lcenter, lwidth, maxConv = Road.find_laneCenterWidth(image_layer, self.sliding_window_width, search_gaps)
        else:
            # startAt lets us start at some intermediate level
            lcenter = startAt[1]
            lwidth = startAt[2]
            lane_centers = startAt[3]

        # Start the full search
        for level in range(startAt[0],num_slices):

            # Verticle slice indices
            idx_start = int(imgH-(level+1)*self.sliding_window_height)
            idx_end = int(imgH-level*self.sliding_window_height)

            image_layer = np.mean(image[idx_start:idx_end,:], axis=0)
            new_lcenter, new_lwidth, maxConv = Road.find_laneCenterWidth(image_layer, self.sliding_window_width, search_gaps)

            if new_lcenter is None or new_lwidth is None:
                continue

            # If the convoution is larger than some value use the result
            if maxConv > 0.2:
                if use_lookahead:
                    if slice_centers[level] < minY:
                        break
                    # if the new values are within the search margin
                    # of the expected values than use them
                    if np.abs(new_lcenter-expected_centers[level]) < self.search_margin and np.abs(new_lwidth-expected_widths[level]) < self.search_margin:
                        weight = 1 if maxConv < 1 else 5
                        lane_centers.append((new_lcenter, slice_centers[level], new_lwidth, weight))

                        #if the right or left lane are out of the image
                        # than stop searching
                        if new_lcenter-new_lwidth/2-self.search_margin/2 < 1 or new_lcenter+new_lwidth/2+self.search_margin/2 > imgW:
                            break
                else:
                    # if the new values are within the search margin of the
                    # old values, then add them to the list
                    if np.abs(new_lcenter-lcenter) < self.search_margin and np.abs(new_lwidth-lwidth) <= self.search_margin:
                        lcenter = new_lcenter
                        lwidth = new_lwidth
                        weight = 1 if maxConv < 1 else 5
                        lane_centers.append((lcenter, imgH-self.sliding_window_height*(level+1/2), lwidth, weight))

                        #if the right or left lane are out of the image
                        # than stop searching
                        if lcenter-lwidth/2-self.search_margin/2 < 1 or lcenter+lwidth/2+self.search_margin/2 > imgW:
                            break

        self.lane_center.addObservation(lane_centers)


    def _window_centroid(self, image, cy, cx):
        # comput the center of mass of a small patch around cx,cy
        lx = int(max(cx-self.search_margin,0))
        rx = int(min(cx+self.search_margin,image.shape[1]))

        by = int(max(cy-self.sliding_window_height/2,0))
        ty = int(min(cy+self.sliding_window_height/2,image.shape[0]))

        window = image[by:ty,lx:rx]
        yc,xc = center_of_mass(window)

        xc += lx
        yc += by
        mass = np.mean(window.flatten())

        return xc,yc,mass

    def _lookahead_line_centroids(self, image):
        # Use the center of mass of the image around the expected
        # values to find new points

        line_centroids = [] # Store the (left_x,left_y) window centroid positions per level

        # Image height
        imgH, imgW = image.shape[0:2]
        fail_count = 0

        # If the expected road is currently very straight, then
        # look-forward along way; however, if the road is curvy, then
        # only look-ahead a maximum of 3 steps.
        if np.abs(self.lane_center.line_fn.c[0]) < 5e-4 and \
            np.abs(self.lane_center.line_fn[1]) < 1:
            minY = 0
        else:
            cp = np.concatenate(self.lane_center.control_points, axis=0)
            minY = np.min(cp[:,1]) - 3*self.sliding_window_height

        num_slices = (int)(imgH/self.sliding_window_height)
        slice_centers = imgH - (np.arange(num_slices)+0.5)*self.sliding_window_height

        # The expected center, width, left, and right values
        x = self.lane_center.line_fn(slice_centers)
        w = self.lane_center.line_width_fn(slice_centers)
        lcx = x-w
        rcx = x+w

        # stop searching when expected left or right lane goes over the edge
        inRange = (lcx+self.search_margin/2 > 1) & \
                    (rcx-self.search_margin/2 < imgW) & \
                    (slice_centers > minY)
        slice_centers = slice_centers[inRange]

        level = 0
        for cy in slice_centers:
            # center of mass and mass of left window
            nlcx,nlcy,lmass = self._window_centroid(image, cy, lcx[level])
            nrcx,nrcy,rmass = self._window_centroid(image, cy, rcx[level])
            lc = (nlcx+nrcx)/2
            lw = (nrcx-nlcx)
            if np.abs(lc-x[level]) < self.search_margin and \
                np.abs(lw/2-w[level]) < self.search_margin and \
                lw > 400 and lw < 850 and \
                (lmass > self.minimum_density or
                rmass > self.minimum_density):

                line_centroids.append((lc, cy, lw, 1))
            level += 1
        else:
            self.lane_center.addObservation(line_centroids)

    def sanity_check(self):
        # If there are not contol points, then re-initialize
        if len(self.lane_center.control_points) == 0:
            self.lane_center.use_lookahead = False

        # If the error is very large than start over
        if self.lane_center.lastErrors[0]>10 or self.lane_center.lastErrors[1]>10:
            self.lane_center.line_coeff.clear()
            self.lane_center.line_width_coeff.clear()
            self.lane_center.use_lookahead = False


    def annotate_image(self, image, control_points=None, fps=None):
        # Lane curvature
        R = self.lane_center.radius_of_curvature(image.shape[0], self.ym_per_pix, self.xm_per_pix)
        # Lane offset
        offset = self.lane_center.line_base_pos(image.shape[0], self.xm_per_pix) - self.xm_per_pix*image.shape[1]/2

        def addText(img, text, pos, size=2):
            img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), 8)
            img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), 2)
            return img

        if np.abs(R) > 1e5:
            image = addText(image, 'Radius of curvature: --- m', (50,60))
        else:
            image = addText(image, 'Radius of curvature: {:.0f} m'.format(R), (50,60))

        image = addText(image, 'Lane center offset: {:.2f} m'.format(offset), (50,130))

        if fps is not None:
            image = addText(image, 'FPS: {:.0f}'.format(fps), (50,180), size=1)

        # Add in lane control points
        if control_points is not None:
            center = control_points[1]
            left = control_points[0]
            right = control_points[2]
            lw = control_points[3]
            for i in range(center.shape[0]):
                cv2.circle(image, (center[i,0,0],center[i,0,1]), lw[i], (0,255,255), -1)
                cv2.circle(image, (left[i,0,0],left[i,0,1]), lw[i], (255,255,0), -1)
                cv2.circle(image, (right[i,0,0],right[i,0,1]), lw[i], (255,255,0), -1)

        return image

    def draw_lane(self, image, toCarView=True, fps=None):

        # Create image to draw the lane in
        lane_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)

        # Get left and right lines
        y = np.arange(0,image.shape[0]).astype(np.float32)
        c = self.lane_center.line_fn(y)
        w = self.lane_center.line_width_fn(y)

        # Cast into good form for cv2.fillPoly()
        lpts = np.array([np.vstack([c-w,y]).T])
        rpts = np.array([np.flipud(np.vstack([c+w,y]).T)])
        pts = np.hstack((lpts,rpts))

        # Draw lane in image
        cv2.fillPoly(lane_img, np.int_([pts]), (0,255,0))

        # Convert to car view
        if toCarView:
            lane_img = self.CarCam.transform_to_cars_view(lane_img)

        # Pad lane image
        # Combine with image
        y_offset = 0
        if lane_img.shape != image.shape:
            y_offset = image.shape[0]-lane_img.shape[0]
            lane_img = np.pad(lane_img, ((y_offset,0), (0,0), (0,0) ), 'constant')

        # Get control points
        if len(self.lane_center.control_points) > 0:
            lw = (5*self.lane_center.control_point_weights()).astype(np.int32)
            c = np.concatenate(self.lane_center.control_points, axis=0).astype(np.int32)

            center = c[:,0:2].astype(np.float32)
            center = center[:,None,:]

            left = np.copy(center)
            right = np.copy(center)
            left[:,0,0] -= c[:,2]/2.
            right[:,0,0] += c[:,2]/2.

            if toCarView:
                center = self.CarCam.transform_to_cars_view(center)
                left = self.CarCam.transform_to_cars_view(left)
                right = self.CarCam.transform_to_cars_view(right)
                center[:,:,1] += y_offset
                left[:,:,1] += y_offset
                right[:,:,1] += y_offset

            control_points = (left, center, right, lw)
        else:
            control_points = None

        annotated_img = cv2.addWeighted(image, 1, lane_img, 0.3, 0)

        return self.annotate_image(annotated_img, control_points=control_points, fps=None)

    def detect_lane(self, image):

        # Adjust the image transformation matrices so that we can just
        # work with the part of the image in the region of intrest
        if self._is_first_image:
            self._is_first_image = False
            # First adjust the view transfromation matrices by chaning
            # the source and destination points
            self._process_img_H = (np.min(self.CarCam.car_points[:,1]).astype(np.int_),
                                    np.max(self.CarCam.car_points[:,1]).astype(np.int_))

            self.CarCam.car_points = self.CarCam.car_points - np.array([0, self._process_img_H[0]],dtype=np.float32)

        tic()
        # Undistort the image
        image = self.CarCam.undistort(image)
        # Crop the image (remove the top)
        proc_img = image[self._process_img_H[0]:self._process_img_H[1],:,:]

        # Binarize the image
        binary_bev = self.Binarizer.binarize(proc_img, self.CarCam)
        # plt.imshow(binary_bev)
        # plt.show()
        # Add the three binary channels and square. This will strongly
        # weight regions where the channels overlap
        binary_bev_flat = np.square(np.sum(binary_bev,axis=2)).astype(np.float32)

        # If we are debuging, then it is useful to see the colored binary
        # image (in birds-eye view) instead of the original image.
        if self.debug:
            # Normalize the image for viewing
            binary_bev = (255*binary_bev/np.max(binary_bev.flatten())).astype(np.uint8)

        # If there are a large number of non-zero pixels in the mask or
        # we are not using look-ahead, then using the slower, but more
        # accurate lane detection method.
        if np.count_nonzero(binary_bev_flat)/binary_bev_flat.size > 0.03 or \
            not self.lane_center.use_lookahead:
            # Note this function still has some lookahead functionality,
            # it is just slower and more robust when there are many mask
            # pixels
            self._initialize_line_centroids(binary_bev_flat)
        else:
            # Use the faster look-ahead method if there are few non-zero
            # pixels
            self._lookahead_line_centroids(binary_bev_flat)

        self.sanity_check()

        compute_time = toc()

        # Draw the detected lane, the control points, and add text
        if self.debug:
            annotated_img = self.draw_lane(binary_bev,False, fps=1/compute_time)
        else:
            annotated_img = self.draw_lane(image, fps=1/compute_time)

        return annotated_img




def generate_lane_finding_example_figure(imgName='./test_images/test5.jpg'):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    road = Road()
    road.CarCam.loadCalibration('cameraCalibrationData.npz')
    road.Binarizer.colorspace = cv2.COLOR_RGB2LAB
    road.lane_center.robustFitting = True

    image = mpimg.imread(imgName)

    image = road.CarCam.undistort(image)
    binary_bev = road.Binarizer.binarize(image, road.CarCam)
    binary_bev_flat = np.square(np.sum(binary_bev,axis=2)).astype(np.float32)


    img_slice = np.mean(binary_bev_flat[int(3*image.shape[0]/5):,:], axis=0)
    search_gaps = np.arange(550,905,50)

    w = road.sliding_window_width
    window = np.ones(w)/(2*w)
    convFilter = lambda gap: np.hstack((window, -np.ones(int(gap - w))/(gap-w), window))
    filterSize = lambda gap: w + gap

    centers = []
    widths = []
    maximums = []
    convolutions = []

    for gap in search_gaps:
        conv = np.convolve(convFilter(gap), img_slice, mode='valid')
        centers.append(np.argmax(conv) + filterSize(gap)/2)
        widths.append(gap)
        maximums.append(np.max(conv))
        convolutions.append(conv)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    img_slice -= np.min(img_slice)
    img_slice /= np.max(img_slice)
    ax.plot(np.arange(img_slice.size), img_slice, 'k-',linewidth=2)
    ax.plot(np.arange(filterSize(650)), convFilter(650)*20+1.2, 'k-',linewidth=2)
    ax.plot(np.arange(filterSize(650)), np.ones(filterSize(650))*1.2, 'k:',linewidth=1)
    ax.plot([w/2,w/2],[1.1,1.7], 'k:',linewidth=1)
    ax.plot([650+w/2,650+w/2],[1.1,1.7], 'k:',linewidth=1)
    ax.text(300, 1.5, 'gap')
    ax.arrow(280, 1.6, w/2-280, 0, width=0.1, shape='right', length_includes_head=True, head_length=25, head_width=0.2, linewidth=0.1)
    ax.arrow(380, 1.55, 650+w/2-380, 0, width=0.1, shape='right', length_includes_head=True, head_length=25, head_width=0.2, linewidth=0.1)

    for count, gap in enumerate(search_gaps):
        x = np.arange(img_slice.size - filterSize(gap) + 1) + filterSize(gap)/2
        ax.plot(x, convolutions[count]+0.2*count+2, color=np.ones(3)*0.5)
        ax.text(x[-1]+50, convolutions[count][1]+0.2*count +2 + 0.02, 'gap = {}'.format(gap))

    x = np.arange(img_slice.size - filterSize(750) + 1) + filterSize(750)/2
    ax.plot(x, convolutions[4]+0.2*4 + 2, 'k-',linewidth=3, color=[0,0,0])

    ax.text(20,3.65,'Convolutions (valid region)', fontstyle='oblique')
    ax.text(750,1.25,'Convolution filter', fontstyle='oblique')
    ax.text(350,0.15,'Mean of image slice', fontstyle='oblique')

    ax.set_xlim([0, img_slice.size])
    ax.get_yaxis().set_visible(False)

    fig.savefig('./output_images/lane_finding_example.png', format='png', transparent=True)

def generate_lookahead_example_figure(imgName='./test_images/test5.jpg'):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    road = Road()
    road.CarCam.loadCalibration('cameraCalibrationData.npz')
    road.Binarizer.colorspace = cv2.COLOR_RGB2LAB
    road.lane_center.robustFitting = True

    image = mpimg.imread(imgName)

    imageud = road.CarCam.undistort(image)
    binary_bev = road.Binarizer.binarize(imageud, road.CarCam)
    binary_bev_flat = np.square(np.sum(binary_bev,axis=2)).astype(np.float32)
    binary_bev = (255*binary_bev/np.max(binary_bev.flatten())).astype(np.uint8)
    # Run an initial detection
    road.detect_lane(image)

    # Run look-ahead detection manually to extract the left and right
    # positions
    w = road.search_margin
    h = road.sliding_window_height

    num_slices = (int)(image.shape[0]/h)
    slice_centers = image.shape[0] - (np.arange(num_slices)+0.5)*h

    # The expected center, width, left, and right values
    center = road.lane_center.line_fn(slice_centers)
    width = road.lane_center.line_width_fn(slice_centers)
    lcx = center-width
    rcx = center+width

    template = np.zeros((binary_bev.shape[0],binary_bev.shape[1],3),dtype=np.uint8)
    line_centroids = []
    for i in range(lcx.shape[0]):
        p1 = np.array((lcx[i]-w, slice_centers[i]-h/2),dtype=np.int32)
        p2 = np.array((lcx[i]+w, slice_centers[i]+h/2),dtype=np.int32)
        cv2.rectangle(template, tuple(p1), tuple(p2), (125, 125, 125), thickness=2)

        p1 = np.array((rcx[i]-w, slice_centers[i]-h/2),dtype=np.int32)
        p2 = np.array((rcx[i]+w, slice_centers[i]+h/2),dtype=np.int32)
        cv2.rectangle(template, tuple(p1), tuple(p2), (125, 125, 125), thickness=2)

        nlcx,nlcy,lmass = road._window_centroid(binary_bev_flat, slice_centers[i], lcx[i])
        nrcx,nrcy,rmass = road._window_centroid(binary_bev_flat, slice_centers[i], rcx[i])
        lc = (nlcx+nrcx)/2
        lw = (nrcx-nlcx)
        if np.abs(lc-center[i]) < road.search_margin and \
            np.abs(lw/2-width[i]) < road.search_margin and \
            lw > 400 and lw < 850 and \
            (lmass > road.minimum_density or
            rmass > road.minimum_density):
            line_centroids.append((int(nlcx), int(nrcx), int(lc), int(slice_centers[i])))

    output = cv2.addWeighted(binary_bev, 1, template, 0.5, 0.0)

    # run look-ahead detection for real and get the equations of the lines
    road.detect_lane(image)

    ys = np.arange(image.shape[0])
    xs = road.lane_center.line_fn(ys)
    pts = np.array([np.vstack([xs,ys]).T])
    cv2.polylines(output, np.int_([pts]), False, (0,0,255), 3)

    for i in range(len(line_centroids)):
        cv2.circle(output, tuple(line_centroids[i][2:4]), 8, (0,0,0), -1)
        cv2.circle(output, (line_centroids[i][0], line_centroids[i][3]), 8, (0,0,0), -1)
        cv2.circle(output, (line_centroids[i][1], line_centroids[i][3]), 8, (0,0,0), -1)

        cv2.circle(output, tuple(line_centroids[i][2:4]), 5, (0,255,255), -1)
        cv2.circle(output, (line_centroids[i][0], line_centroids[i][3]), 5, (255,255,0), -1)
        cv2.circle(output, (line_centroids[i][1], line_centroids[i][3]), 5, (255,255,0), -1)

    mpimg.imsave('./output_images/lookahead_example.jpg', output)


def generate_pipeline_figure(imgName='./test_images/test5.jpg'):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    road = Road()
    road.CarCam.loadCalibration('cameraCalibrationData.npz')
    road.Binarizer.colorspace = cv2.COLOR_RGB2LAB
    road.lane_center.robustFitting = True

    image = mpimg.imread(imgName)
    binary_bev = road.Binarizer.binarize(image, road.CarCam)
    binary_bev_flat = np.square(np.sum(binary_bev,axis=2)).astype(np.float32)
    imageu = road.CarCam.undistort(image)
    img_lab = cv2.cvtColor(imageu, cv2.COLOR_RGB2LAB)

    fig = plt.figure(figsize=(7,6))
    ax1 = fig.add_axes([0.35,0.75,0.3,0.2])

    ax2 = fig.add_axes([0.01,0.5,0.3,0.2])
    ax3 = fig.add_axes([0.35,0.5,0.3,0.2])
    ax4 = fig.add_axes([0.69,0.5,0.3,0.2])

    ax5 = fig.add_axes([0.01,0.25,0.3,0.2])
    ax6 = fig.add_axes([0.35,0.25,0.3,0.2])
    ax7 = fig.add_axes([0.69,0.25,0.3,0.2])

    ax8 = fig.add_axes([0.35,0.0,0.3,0.2])

    ax1.imshow(imageu)
    ax1.axis('off')
    ax1.set_title('Original (undistorted) image')

    ax2.imshow(img_lab[:,:,0],cmap='gray',vmin=0,vmax=255)
    ax2.axis('off')
    ax2.set_title('L')

    ax3.imshow(img_lab[:,:,2],cmap='gray',vmin=0,vmax=255)
    ax3.axis('off')
    ax3.set_title('b')

    ax4.imshow(0.5*(img_lab[:,:,0]+img_lab[:,:,2]),cmap='gray',vmin=0,vmax=255)
    ax4.axis('off')
    ax4.set_title('(L+b)/2')

    ax5.imshow(binary_bev[:,:,0],cmap='gray',vmin=0,vmax=1)
    ax5.axis('off')
    ax5.set_title('bev() > 210')

    ax6.imshow(binary_bev[:,:,1],cmap='gray',vmin=0,vmax=1)
    ax6.axis('off')
    ax6.set_title('F(bev(d/dx) ≷ ±20)')

    ax7.imshow(binary_bev[:,:,2],cmap='gray',vmin=0,vmax=1)
    ax7.axis('off')
    ax7.set_title('F(bev(d/dx) ≷ ±25)')

    ax8.imshow(binary_bev_flat, cmap='gray')
    ax8.axis('off')
    ax8.set_title('Combined (Σ²)')

    from matplotlib.patches import ConnectionPatch
    def add_arrow(p1,p2,ax1,ax2):
        con = ConnectionPatch(xyA=p1, xyB=p2, coordsA="axes fraction", coordsB="axes fraction",
                          axesA=ax1, axesB=ax2, arrowstyle='-|>')
        ax2.add_artist(con)

    add_arrow((0,0),(1,1),ax1,ax2)
    add_arrow((0.6,0),(0.6,1),ax1,ax3)
    add_arrow((1,0),(0,1),ax1,ax4)

    add_arrow((0.95,0),(0.95,1),ax2,ax5)
    add_arrow((0.95,0),(0.95,1),ax3,ax6)
    add_arrow((0.05,0),(0.05,1),ax4,ax7)

    add_arrow((1,0),(0,1),ax5,ax8)
    add_arrow((0.9,0),(0.9,1),ax6,ax8)
    add_arrow((0,0),(1,1),ax7,ax8)

    fig.savefig('./output_images/image_pipeline_example.png', format='png', transparent=True)

if __name__ == '__main__':
    generate_lane_finding_example_figure()
    generate_lookahead_example_figure()
    generate_pipeline_figure()

    # road = Road()
    # road.CarCam.loadCalibration('cameraCalibrationData.npz')
    # road.lane_center.robustFitting = True
    # road.lane_center.history_size = (5,3,2)
    # road.debug = True
# image = mpimg.imread('./test_images/test5.jpg')
# road.Binarizer.colorspace = cv2.COLOR_RGB2LAB
# #
# # plt.imshow(image)
# # plt.show()
# #
# plt.imshow(road.detect_lane(image))
# plt.show()


    # cap = cv2.VideoCapture('./challenge_video.mp4')
    #
    # counter = 0
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         imgOut = road.detect_lane(frame)
    #         cv2.imshow('frame',imgOut)
    #
    #         # break
    #         # counter += 1
    #         # # print(counter)
    #         # if counter > 20:
    #         #     break
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
