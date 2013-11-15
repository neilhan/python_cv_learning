#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The reusable functions for learning.

The reusable functions for learning.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import random
import cv
import cv2
import numpy as np
import ava.utl


def show_image_wait(image):
    # if isinstance(image, (cv.cvmat)):
    cv.NamedWindow("The window", cv.CV_WINDOW_AUTOSIZE)

    cv.ShowImage("The window", image)
    cv.WaitKey(10000)


# cv2
def show_image_wait_2(image, waitMS=10000):
    # if isinstance(image, (cv.cvmat)):
    cv2.imshow("The window", image)
    cv2.waitKey(waitMS)


def color_reduce(image, div = 64):
    if image.channels == 1:
        for y in range(image.rows):
            for x in range(image.cols):
                image[y, x] = image[y, x] // div * div + div // 2
    elif image.channels == 3:
        for y in range(image.rows):
            for x in range(image.cols):
                b, g, r = image[y, x]
                b = int(b) / div * div + div/2
                g = int(g) / div * div + div/2
                r = int(r) / div * div + div/2
                image[y, x] = [b, g, r]


@ava.utl.time_this
def color_reduce_2(image, div = 64):
    """image:np.array, div:int of 2^n, -> color reduced image as np.array """
    assert isinstance(image, np.ndarray), 'image needs to be a np.ndarray'
    theAdd = div // 2
    image /= div
    image *= div
    image += theAdd
    return image


@ava.utl.time_this
def color_reduce_vector_2(image, div = 64):
    assert isinstance(image, np.ndarray), 'image needs to be a np.ndarray'
    def Reduce(a, d):
        return a // d * d + d // 2
    vectReduce = np.vectorize(Reduce, otypes=[image.dtype])
    image2 = vectReduce(image, div)
    return image2


