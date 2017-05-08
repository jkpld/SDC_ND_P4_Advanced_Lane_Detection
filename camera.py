import cv2
import glob
import numpy as np
import matplotlib.image as mpimg

# Class Camera will internally hold the camera matrix and distortion
# coefficients as well as transformation matricies for converting between
# birdseye view and the car view. The class has several methods:
#   calibrateCamera : will take in folder name containing images (named
#     calibration*.jpg) of checker boards and the size of the checker
#     board. The function will then compute the camera matrix and
#     distortion coefficients.
#   undistort : will take in an image and apply the distortion correction.
#   transform_to_birdseye_view :
#   transform_to_cars_view :
#   saveCalibration : save the camera matrix and distortion coefficients
#   loadCalibration : load a camera matrix and distortion coefficients
class Camera:
    def __init__(self, filename=None):

        # Camera calibration
        self.cameraMatrix = None
        self.distCoeffs = None
        self._objpoints = []
        self._imgpoints = []

        self.loadCalibration(filename)

        # Perspective transform
        self._car_points = np.float32([[200,720],[585,455],[695,455],[1080,720]])
        self._birdseye_points = np.float32([[300,720],[300,0],  [980,0], [980,720]])

        self._in_height = 720
        self._out_height = 720

        self._M = cv2.getPerspectiveTransform(self._car_points, self._birdseye_points)
        self._Minv = np.linalg.inv(self._M)

    @property
    def car_points(self):
        return self._car_points

    @car_points.setter
    def car_points(self, pts):
        self._car_points = pts
        if pts.shape == self._birdseye_points.shape:
            self._M = cv2.getPerspectiveTransform(self._car_points, self._birdseye_points)
            self._Minv = np.linalg.inv(self._M)
            self._in_height = np.max(pts[:,1])
        else:
            print('Warning! Number of car-view points and birds-eye-view points are not the same. Cannot update transformation matrices.')

    @property
    def birdseye_points(self):
        return self._birdseye_points

    @birdseye_points.setter
    def birdseye_points(self, pts):
        self._birdseye_points = pts
        if pts.shape == self._car_points.shape:
            self._M = cv2.getPerspectiveTransform(self._car_points, self._birdseye_points)
            self._Minv = np.linalg.inv(self._M)
            self._out_height = np.max(pts[:,1])
        else:
            print('Warning! Number of car-view points and birds-eye-view points are not the same. Cannot update transformation matrices.')

    def calibrateCamera(self, folder, chessBoardSize=(9,6)):
        imageNames = glob.glob('{}/calibration*.jpg'.format(folder))

        objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessBoardSize[0],0:chessBoardSize[1]].T.reshape(-1,2)

        # Read in images and create the image points and object points
        # arrays
        for imageName in imageNames:
            # read in image
            img = mpimg.imread(imageName) # RGB image

            # convert to gray scale
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessBoardSize, None)

            # if corners are found, add points to list
            if ret == True:
                self._imgpoints.append(corners)
                self._objpoints.append(objp)

        # Create calibration
        # Note, this line makes the assumption that all images are the same size
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self._objpoints, self._imgpoints, gray.shape[::-1], None, None)

        # Save calibration matrix and distortion coeffs.
        self.cameraMatrix = mtx
        self.distCoeffs = dist

    def undistort(self, img):
        if self.cameraMatrix is None:
            raise ValueError('cameraMatrix is None, must first calibrate the camera before running undistort.')
        else:
            return cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

    def transform_to_birdseye_view(self, img, shape=None):
        if img.ndim == 3:
            if img.shape[2]==2 and img.shape[1]==1:
                # Assume set of points
                return cv2.perspectiveTransform(img, self._M)

        if shape is None:
            imshape = (img.shape[1], self._out_height)
        else:
            imshape = shape
        return cv2.warpPerspective(img, self._M, imshape)

    def transform_to_cars_view(self, img, shape=None):

        if img.shape[2]==2 and img.shape[1]==1:
            # Assume set of points
            return cv2.perspectiveTransform(img, self._Minv)
        else:
            # Assume image
            if shape is None:
                imshape = (img.shape[1], self._in_height)
            else:
                imshape = shape
            return cv2.warpPerspective(img, self._Minv, imshape)

    def saveCalibration(self, filename):
        np.savez(filename, cameraMatrix=self.cameraMatrix,
                           distCoeffs=self.distCoeffs,
                           objpoints=self._objpoints,
                           imgpoints=self._imgpoints)

    def loadCalibration(self, filename):
        if filename is None:
            self.cameraMatrix = None
            self.distCoeffs = None
            self._objpoints = []
            self._imgpoints = []
        else:
            try:
                with np.load(filename) as data:
                    self.cameraMatrix = data['cameraMatrix']
                    self.distCoeffs = data['distCoeffs']
                    self._objpoints = data['objpoints']
                    self._imgpoints = data['imgpoints']
            except:
                print('Error loading file')
                self.cameraMatrix = None
                self.distCoeffs = None
                self._objpoints = []
                self._imgpoints = []



# If running the program the file itself, then create an example image
# showing the distortion correction
if __name__ == '__main__':

    def addText(img, text, pos):
        img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 8)
        img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        return img

    # Directory containing calibration images
    calibrationImageDir = './camera_cal'

    # Create Camera object
    CarCam = Camera()
    CarCam.calibrateCamera(folder=calibrationImageDir, chessBoardSize=(9,6))

    # Read in text calibration image and apply correction
    img = mpimg.imread('./camera_cal/calibration16.jpg')
    img_undistorted = CarCam.undistort(img)

    # Combine images, add text, and save the image
    example_image = np.concatenate((img, img_undistorted), axis=1)
    example_image = addText(example_image, 'Original image', (50, 100))
    example_image = addText(example_image, 'Undistorted image', (1330,100))

    mpimg.imsave('./output_images/undistorted_calibration16.jpg', example_image)

    # Read in text road image and apply correction
    img = mpimg.imread('./test_images/test5.jpg')
    img_undistorted = CarCam.undistort(img)

    # Combine images, add text, and save the image
    example_image = np.concatenate((img, img_undistorted), axis=1)
    example_image = addText(example_image, 'Original image', (50, 100))
    example_image = addText(example_image, 'Undistorted image', (1330,100))

    mpimg.imsave('./output_images/undistorted_test5.jpg', example_image)

    # Read in text road image and apply correction
    bev = CarCam.transform_to_birdseye_view(img_undistorted)

    cv2.polylines(img_undistorted, np.int32([CarCam.car_points]), True, (255,0,0), 3)
    cv2.polylines(bev, np.int32([CarCam.birdseye_points]), True, (255,0,0), 3)

    # Combine images, add text, and save the image
    example_image = np.concatenate((img_undistorted, bev), axis=1)
    example_image = addText(example_image, 'Undistorted image', (50, 100))
    example_image = addText(example_image, 'Birds-eye view', (1330,100))

    mpimg.imsave('./output_images/birdseyeView_test5.jpg', example_image)
