#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Testing split image? I don't remember what I was doing this for. """

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import sys
import cv2
cv2.namedWindow('WorkAroundTheCoreDump')
cv2.destroyWindow('WorkAroundTheCoreDump')
import numpy as np
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl

def getHistogram(image):
    # how many image, channels, mask, result, ?D histogram,
    # number of bins, pixel value range
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


@ava.utl.time_this
def main(argv=None):
    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:, :, 0] = h
    hsv_map[:, :, 1] = s
    hsv_map[:, :, 2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HLS2BGR)

    if argv == None:
        argv = sys.argv

    image1 = cv2.imread("../images/pic3.jpg", 1)
    print(type(image1))
    print("image.shape:", image1.shape)
    print('image:', image1)
    print('image dtype:', image1.dtype)

    # split image
    imageHsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist( [imageHsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    hist2 = np.clip(hist * 0.005 * 100, 0, 1)
    vis = hsv_map * hist2[:, :, np.newaxis] / 255.0

    cv2.imshow('image1', image1)
    cv2.imshow('hist', vis)

    cv2.waitKey()


if __name__ == "__main__":
    main()
