from road import Road
from moviepy.editor import VideoFileClip
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Setup of road lane detector
road = Road()
road.CarCam.loadCalibration('cameraCalibrationData.npz')
road.Binarizer.colorspace = cv2.COLOR_RGB2LAB
road.lane_center.robustFitting = True
road.lane_center.history_size = (5,3,2)


# Process videos
video_names = ["project_video", "challenge_video", "harder_challenge_video"]

for video in video_names:
    for debug in (True, False):
        # Get video input and output names
        sub_name = "_debug2" if debug else "_processed2"
        video_out = video + sub_name + ".mp4"
        video_in = video + ".mp4"

        print('Processing {} ({})'.format(video_in, debug))

        # Set debugging state and process the video
        road.reset()
        road.debug = debug
        clip1 = VideoFileClip(video_in)
        processed_clip = clip1.fl_image(road.detect_lane)
        processed_clip.write_videofile(video_out, audio=False)

# Process test images
input_folder = './test_images/'
output_folder = './output_images/'
imageNames = glob.glob(input_folder + '*.jpg')

for imageName in imageNames:
    img = mpimg.imread(imageName)

    for debug in (True, False):
        sub_name = "_debug" if debug else "_processed"
        output_name = output_folder + imageName[len(input_folder):-4] + \
            sub_name + '.jpg'

        road.reset()
        road.debug = debug
        processed_img = road.detect_lane(img)
        mpimg.imsave(output_name, processed_img)
