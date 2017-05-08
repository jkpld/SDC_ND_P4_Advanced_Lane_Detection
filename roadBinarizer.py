import numpy as np
import cv2
import matplotlib.pyplot as plt


from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc():
    return (time() - _tstart_stack.pop())

# RoadBinarizer will take in an image of a road and binarize it for
# lane line detection and transform to birds-eye view (if requested):
#
# The colorspace of the original image will be converted using
#   the transformation given by colorspace
# New image channels will be created by averaging the Transformed
#   image over the indices given by channels. The number of new
#   channels will be len(channels)
# Each new channel will be left as a color channel if gradient is
#   0; if gradient is 1, then the x sobel derivative will be taken;
#   if gradient is 2, then the y sobel derivative will be taken.
#   If a derivative is taken, then the end result will be nomalized
#   to have a maximum value of 255.
# The threshold ranges in thresholds (min, max) will be applied to
#   each channel.
# The output will be a image with the number of channels given by
#   len(channels).
#
# Example:
# binarizer = RoadBinarizer(colorspace=cv2.COLOR_BGR2LAB,
#                           channels=[[0], [2], [0,2]],
#                           gradient=[0,1,1],
#                           thresholds=[(210,255), (20,255), (25,255)]))
#
# binary = binarizer.binarize(img)
#
# binary will have three channels:
# - The first channel will be the L channel thresholed between 210 and 255.
# - The second channel will be the x-gradient of the b channel thresholed
# between 20 and 255. (Note that the x-gradient is filted to look for
# light on dark lines)
# - The third channel will be the x-gradient of the L channel plus the b
# channel thresholded between 25 and 255.

class RoadBinarizer():
    def __init__(self, colorspace=cv2.COLOR_BGR2LAB,
                    channels=[[0], [2], [0,2]],
                    gradient=[0,1,1],
                    thresholds=[(210,255), (20,255), (25,255)]):

        self.colorspace = colorspace
        self.channels = channels
        self.gradient = gradient
        self.thresholds = thresholds
        self.gradient_dilate_kernal = (2,8)
        self.num_dilation_iterations = 3

    def filter_gradient(self, img, threshold):
        # This function will threshold the gradient at +- threshold. It
        # will then shift the negative mask to the left and the positive
        # mask to the right and intersect the two masks. This will only
        # allow light on dark lines to pass through.
        #
        # Credit for this idea comes from balancap:
        # https://github.com/balancap/SDC-Advanced-Lane-Finding
        mask_neg = (img < -threshold).astype(np.uint8)
        mask_pos = (img > threshold).astype(np.uint8)

        dilate_kernel = self.gradient_dilate_kernal
        mid = dilate_kernel[1] // 2
        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, dtype=np.uint8)
        kernel[:, 0:mid] = 0
        dmask_neg = cv2.dilate(mask_neg, kernel, iterations=self.num_dilation_iterations) > 0.
        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, np.uint8)
        kernel[:, mid:] = 0
        dmask_pos = cv2.dilate(mask_pos, kernel, iterations=self.num_dilation_iterations) > 0.
        dmask = (dmask_pos * dmask_neg).astype(np.uint8)

        return dmask

    @staticmethod
    def grad(img, k):
        dG = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        dG = 255*dG/np.max(dG)
        return dG

    def binarize(self, img_orig, carcam=None):
        # Convert to new colorspace
        img = cv2.cvtColor(img_orig, self.colorspace).astype(np.float)

        # Iterate over each new channel to make
        for i in range(len(self.channels)):

            # Compute new channel
            if len(self.channels[i]) > 1:
                tmpImg = np.mean(img[:,:,self.channels[i]],axis=2)
            else:
                tmpImg = img[:,:,self.channels[i]]

            # Take gradient if requested
            if self.gradient[i] == 1:
                tmpImg = RoadBinarizer.grad(tmpImg, 7) # Take the derivative in x
                # The gradient can be filtered to look for light on dark
                # lines; however, this correction should take place in the
                # birds-eye view.
                if carcam is not None:
                    tmpImg = carcam.transform_to_birdseye_view(tmpImg)
                # Threshold and filter the gradient
                tmpImg = self.filter_gradient(tmpImg, self.thresholds[i][0])
            else:
                # otherwise, go to birds eye view and threshold
                if carcam is not None:
                    tmpImg = carcam.transform_to_birdseye_view(tmpImg)
                tmpImg = ((tmpImg >= self.thresholds[i][0]) & (tmpImg <= self.thresholds[i][1])).astype(np.uint8)

            # Initialize array for output
            if i == 0:
                binarized = np.zeros((tmpImg.shape[0],tmpImg.shape[1],len(self.channels)))

            binarized[:,:,i] = np.squeeze(tmpImg)

        return binarized



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2
    from camera import Camera

    CarCam = Camera('cameraCalibrationData.npz')
    if CarCam.cameraMatrix is None:
        calibrationImageDir = './camera_cal'
        CarCam.calibrateCamera(calibrationImageDir, chessBoardSize=(9,6))

    # Read in image and undistort
    # image = mpimg.imread('./test_images/straight_lines2.jpg')
    image = mpimg.imread('./test_images/test5.jpg')
    image = CarCam.undistort(image)

    # Binarize image and save results
    roadBinarizer = RoadBinarizer()
    roadBinarizer.colorspace = cv2.COLOR_RGB2LAB
    binarized = roadBinarizer.binarize(image)

    # Plot the result ----------------------------------------------------
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original (undistorted) Image', fontsize=20)

    ax2.imshow(binarized)
    ax2.set_title('Binarization', fontsize=20)

    ax3.imshow(np.sum(binarized,axis=2)>0.5, cmap='gray')
    ax3.set_title('Binarized Result', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.)

    f.savefig('./output_images/binarization_results.png', format='png', transparent=True)
    # --------------------------------------------------------------------

    # Transform to birds-eye view
    binarized = (np.sum(binarized,axis=2)>0.5).astype(np.float32)
    birdseyebinarized = CarCam.transform_to_birdseye_view(binarized)
    cv2.imwrite('./output_images/binarized_test5.jpg', 255*binarized)
    cv2.imwrite('./output_images/birdseye_binarized_test5.jpg', 255*birdseyebinarized)

    binarized = np.dstack((binarized,binarized,binarized))
    birdseyebinarized = CarCam.transform_to_birdseye_view(binarized)

    # Plot the result ----------------------------------------------------
    # Draw source and destination points
    cv2.polylines(binarized, np.int32([CarCam.car_points]), True, (1,0,0), 3)
    cv2.polylines(birdseyebinarized, np.int32([CarCam.birdseye_points]), True, (1,0,0), 3)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    f.tight_layout()

    tmp = ax1.imshow(binarized,cmap='gray')
    ax1.set_title('Original view', fontsize=20)
    # f.colorbar(tmp, ax=ax1)

    tmp = ax2.imshow(birdseyebinarized,cmap='gray')
    ax2.set_title('Birds-eye view', fontsize=20)
    # f.colorbar(tmp, ax=ax2)

    plt.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.)


    f.savefig('./output_images/birdsEye_results.png', format='png', transparent=True)
    # --------------------------------------------------------------------
