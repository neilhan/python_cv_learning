#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

# import pdb; pdb.set_trace() ; # debugging--------------------------------------
import sys
import logging

import cv2
cv2.namedWindow('GetArroundASegmentationFailure', 0)
cv2.destroyWindow('GetArroundASegmentationFailure')
import numpy as np
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl


@ava.utl.time_this
def main(argv=None):
    ava.utl.setup_logging()
    logger = logging.getLogger('RunMe.main')

    image1 = cv2.imread("../pic1.jpg", flags=0)
    logger.debug(type(image1))
    logger.debug("image.shape: " + str(image1.shape))
    logger.debug('image dtype: ' + str(image1.dtype))
    pltHist(image1)
    cv2Hist(image1)
    npHist(image1)
    plotHist()
    cv2_draw()


@ava.utl.time_this
def pltHist(image):
    plt.hist(image.ravel(), 256, [0, 256])  # plt.show()


@ava.utl.time_this
def cv2Hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])


@ava.utl.time_this
def npHist(image):
    hist, bins = np.histogram(image, 256, [0, 256])


def plotHist():
    image1 = cv2.imread('../pic1.jpg', flags=1)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image1], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


@ava.utl.time_this
def cv2_draw():
    logger = logging.getLogger('RunMe.cv2Draw')
    image1 = cv2.imread('../pic1.jpg', flags=1)
    h = np.zeros((300, 256, 3))

    bins = np.arange(256).reshape(256, 1)
    colors = [(256, 0, 0), (0, 256, 0), (0, 0, 256)]
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image1], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        normHist = np.int32(np.around(hist))
        tempHist = normHist.flat
        print(type(tempHist))
        for j in range(hist.size):
            if tempHist[j] < 0 or tempHist[j] > 255:
                print(i, j, tempHist[j])

        points = np.column_stack((bins, normHist))
        points = np.array(points)
        # cv2 polyline give p.checkVector error
        cv2.polylines(h, [points], 1, color)
        # ava.utl.polyline(h, points, color)

    # import pdb; pdb.set_trace() ; # debugging--------------------------------------
    h = np.flipud(h)

    cv2.imshow('colorhist', h)
    cv2.waitKey(0)
    # cv.ShowImage('colorhist', image1)
    # cv.WaitKey(0)


if __name__ == "__main__":
    main()
