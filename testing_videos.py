# USAGE
# python realtime_stitching.py

# import the necessary packages
from __future__ import print_function

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS

from pyimagesearch.basicmotiondetector import BasicMotionDetector
from pyimagesearch.panorama import Stitcher
fps = FPS().start()
rightStream = VideoStream('test1.mp4')
leftStream = VideoStream('test2.mp4')


print('Getting the Video')
stitcher = Stitcher()
motion = BasicMotionDetector(minArea=15000)
total = 0
while True:
    left = leftStream.read()
    right = rightStream.read()

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


    print('Video Stitching Successfully')
    cv2.imshow("Result", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
fps.stop()
print('elapsed time:'.format(fps.elapsed()))
print('approx. FPS'.format(fps.fps()))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
leftStream.stop()
rightStream.stop()
