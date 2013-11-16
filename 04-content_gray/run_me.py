#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

# import ipdb; ipdb.set_trace() ; # debugging--------------------------------------
import sys
import logging

import numpy as np
import scipy.sparse as sp
import cv2
cv2.namedWindow('GetArroundASegmentationFailure', 0)
cv2.destroyWindow('GetArroundASegmentationFailure')
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl


class Histogram1D:
    def __init__(self):
        print("Histogram1D created.")

    def getHistogram(self, image):
        # how many image, channels, mask, result, ?D histogram,
        # number of bins, pixel value range
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv

    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    # read image
    src_img = cv2.imread('../images/waves.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    src_img_bgr = cv2.imread('../images/waves.jpg')
    src_img_hsv = cv2.cvtColor(src_img_bgr, cv2.COLOR_BGR2HSV)
    src_img_cp = src_img_bgr
    ava.cv.utl.show_image_wait_2(src_img) # ---------
    # draw rectangle
    cv2.rectangle(src_img_bgr, (360, 144), (400, 194), (255,200,100))
    ava.cv.utl.show_image_wait_2(src_img_bgr) # --------

    # get the cloud hist
    hist_sample = cv2.calcHist([src_img_hsv[144:195,360:401]], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist_img = cv2.calcHist([src_img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # normalize the sample histogram
    cv2.normalize(hist_sample, hist_sample, 0, 255, cv2.NORM_MINMAX)
    distance = cv2.calcBackProject([src_img_hsv], [0, 1], hist_sample, [0, 180, 0, 256], 1)

    # show the distance
    ava.cv.utl.show_image_wait_2(distance) # ------------

    # convolute with circular
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(distance, -1, kernel, distance)
    # show the smoothed distance
    ava.cv.utl.show_image_wait_2(distance) # ------------

    # threshold
    ret, thresh = cv2.threshold(distance, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge([thresh, thresh, thresh])

    # do the bitwise_and
    result = cv2.bitwise_and(src_img_cp, thresh)
    result = np.vstack([src_img_cp, thresh, result])
    ava.cv.utl.show_image_wait_2(result)


if __name__ == "__main__":
    main()
