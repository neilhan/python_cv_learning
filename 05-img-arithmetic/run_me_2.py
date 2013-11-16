#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import sys
import ava.cv.utl
import random
import numpy as np

def main( argv = None ):
    if argv is None:
        argv = sys.argv

    # test_with_gray()
    print('---------------color testing starts-------------------')
    # test_with_color()
    test_with_color_2()


def test_with_color( argv = None ):

    image1 = cv2.imread("../images/pic1.jpg")
    image2 = cv2.imread("../images/pic2.jpg")
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    print("image1 type:", image1.dtype)
    print("image.shape:", image1.shape)

    print('image1 shape:', image1.shape)
    print('image2 shape:', image2.shape)

    image1ROI = image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2]
    image2ROI = image2[(image2.shape[0] // 2):image2.shape[0], (image2.shape[1] // 2):image2.shape[1]]

    print('calculating...')
    image1ROI = cv2.addWeighted(image1ROI, 0.3, image2ROI, 0.7, 0.0)
    print("showing image")
    ava.cv.utl.show_image_wait_2(image1)
    image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2] = image1ROI
    ava.cv.utl.show_image_wait_2(image1)


def test_with_color_2( argv = None ):

    image1 = cv2.imread("../images/pic1.jpg")
    image2 = cv2.imread("../images/pic2.jpg")
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image1ROI = image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2]
    image2ROI = image2[(image2.shape[0] // 2):image2.shape[0], (image2.shape[1] // 2):image2.shape[1]]

    print('calculating...')
    image1ROI = image1ROI * 0.4 + image2ROI * 0.7
    print("showing image")
    ava.cv.utl.show_image_wait_2(image1)
    image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2] = image1ROI
    ava.cv.utl.show_image_wait_2(image1)



def test_with_gray( argv = None ):

    image1 = cv2.imread("../images/pic1.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("../images/pic2.jpg", cv2.IMREAD_GRAYSCALE)
    # convert to gray
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    print("image1 type:", image1.dtype)
    print("image.shape:", image1.shape)

    print('image1 shape:', image1.shape)
    print('image2 shape:', image2.shape)

    image1ROI = image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2]
    image2ROI = image2[(image2.shape[0] // 2):image2.shape[0], (image2.shape[1] // 2):image2.shape[1]]

    print('calculating...')
    image1ROI = cv2.addWeighted(image1ROI, 0.3, image2ROI, 0.7, 0.0)
    print("showing image")
    ava.cv.utl.show_image_wait_2(image1)
    image1[0:image2.shape[0] // 2, 0:image2.shape[1] // 2] = image1ROI
    ava.cv.utl.show_image_wait_2(image1)

    image1_copy = cv2.imread('../images/pic2.jpg')
    image1_strange = image1_copy
    mapping = np.arange(255, -1, -1)

    image1_strange[:, :, 0] = mapping[image1_strange[:, :, 0]]
    ava.cv.utl.show_image_wait_2(image1_strange)

    image1_strange[:, :, 1] = mapping[image1_strange[:, :, 1]]
    ava.cv.utl.show_image_wait_2(image1_strange)

    image1_strange[:, :, 2] = mapping[image1_strange[:, :, 2]]
    ava.cv.utl.show_image_wait_2(image1_strange)


if __name__ == "__main__":
    main()
