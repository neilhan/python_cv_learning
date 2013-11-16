#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code is a test on a few different ways for threshold

"""

from __future__ import absolute_import, division, \
    print_function, unicode_literals

# import ipdb; ipdb.set_trace() ; # debugging--------------------------------------
import sys
import logging

import cv2
cv2.namedWindow('WorkAroundTheCoreDump')
cv2.destroyWindow('WorkAroundTheCoreDump')
import numpy as np
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv

    ava.utl.setup_logging()
    logger = logging.getLogger('RunMe.main')

    image1 = cv2.imread("../images/pic1.jpg", 1)
    print(type(image1))
    print("image.shape:", image1.shape)
    print('image:', image1)
    print('image dtype:', image1.dtype)

    # split image
    image1BW = np.zeros((image1.shape[0], image1.shape[1]), dtype=image1.dtype)
    # b = cv2.split(a)[0]
    image1BW[:, :] = image1[:, :, 0]
    print("type(image1BW)", type(image1BW))
    print("image1BW.shape:", image1BW.shape)
    ava.cv.utl.show_image_wait_2(image1BW)

    # threshold
    image2BW = np.zeros((image1.shape[0], image1.shape[1]), dtype=image1.dtype)
    cv2.threshold(image1BW, 100, 200, cv2.THRESH_TOZERO, image2BW)
    print(type(image2BW))
    ava.cv.utl.show_image_wait_2(image2BW)

    ret, thresh1 = cv2.threshold(image1BW, 127, 255, cv2.THRESH_BINARY)
    # import ipdb; ipdb.set_trace() ; # debugging--------------------------------------
    ret, thresh2 = cv2.threshold(image1BW, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image1BW, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image1BW, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image1BW, 127, 155, cv2.THRESH_TOZERO_INV)

    thresh = ['image1BW', 'thresh1', 'thresh2',
              'thresh3', 'thresh4', 'thresh5']

    for i in xrange(6):
        plt.subplot(2, 3, i + 1), plt.imshow(eval(thresh[i]), 'gray')
        plt.title(thresh[i])

    plt.show()

    print('AdaptiveThreshold::::::::::')
    # adaptive threshold
    image1BWBlur = cv2.medianBlur(image1BW, 15)

    ret, thresh1 = cv2.threshold(image1BWBlur, 127, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(image1BWBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh3 = cv2.adaptiveThreshold(image1BWBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    plt.subplot(2, 2, 1), plt.imshow(image1BWBlur, 'gray')
    plt.title('input')
    plt.subplot(2, 2, 2), plt.imshow(thresh1, 'gray')
    plt.title('Global threshold')
    plt.subplot(2, 2, 3), plt.imshow(thresh2, 'gray')
    plt.title('Adaptive mean threshold')
    plt.subplot(2, 2, 4), plt.imshow(thresh3, 'gray')
    plt.title('Adaptive gaussian threshold')

    plt.show()

    # Otsu's Binarization
    ret1, thresh1 = cv2.threshold(image1BW, 127, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(image1BW, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #blur gaussian filter
    image1BWBlur = cv2.GaussianBlur(image1BW, (15, 15), 0)
    ret3, thresh3 = cv2.threshold(image1BWBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['image1BW', 'histogram1', 'thresh1',
              'image1BW', 'histogram2', 'thresh2',
              'image1BWBlur', 'histogram3', 'thresh3']
    for i in xrange(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(eval(titles[i * 3]), 'gray')
        plt.title(titles[i * 3])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(eval(titles[i * 3]).ravel(), 256)
        plt.title(titles[i * 3 + 1])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(eval(titles[i * 3 + 2]), 'gray')
        plt.title(titles[i * 3 + 2])

    plt.show()


if __name__ == "__main__":
    main()
