# USAGE
# python realtime_stitching.py

# import the necessary packages
from __future__ import print_function

import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

from pyimagesearch.basicmotiondetector import BasicMotionDetector
from pyimagesearch.panorama import Stitcher

# rightStream = VideoStream('test1.mp4')
# leftStream = VideoStream('test2.mp4')

rightStream = cv2.VideoCapture('15.mp4')
leftStream = cv2.VideoCapture('16.mp4')
fps_start_time = 0
fps = 0

print('Getting the Video')
stitcher = Stitcher()
motion = BasicMotionDetector(minArea=15000)
total = 0
while True:
    fps_start_time = time.time()
    ret, right = rightStream.read()
    ret, left = leftStream.read()

    # cv2.imshow('Right Video', right)
    # cv2.imshow('Left Video', left)
    # cv2.waitKey(10)

    # resize the frames
    left = imutils.resize(left, width=400)
    right = imutils.resize(right, width=400)

    # stitch the frames together to form the panorama
    # IMPORTANT: you might have to change this line of code
    # depending on how your cameras are oriented; frames
    # should be supplied in left-to-right order
    result = stitcher.stitch([left, right])

    # no homograpy could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break

    # convert the panorama to grayscale, blur it slightly, update
    # the motion detector
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    locs = motion.update(gray)
    if total > 32 and len(locs) > 0:
        # initialize the minimum and maximum (x, y)-coordinates,
        # respectively
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    print('time taken: {0} seconds'. format(time_diff))
    # fps = 2 / (time_diff)
    # print('fps value', fps)
    # fps_start_time = fps_end_time
    # fps_text = "FPS: {:.2f}".format(fps)
    # cv2.putText(right, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    # cv2.putText(left, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    print('Video Stitching Successfully')
    cv2.imshow("Result", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    cv2.waitKey(20)
#    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
#    if key == ord("q"):
#        break
# do a bit of cleanup

print('elapsed time:'.format(fps.elapsed()))
print('approx. FPS'.format(fps.fps()))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
leftStream.stop()
rightStream.stop()
