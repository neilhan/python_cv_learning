#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This has salt, and color reducing tests.

This has salt, and color reducing tests.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
cv2.namedWindow('MainWindow')
cv2.destroyWindow('MainWindow')
import sys
import numpy as np
import ava.cv.utl
import random



def salt(image, n):
    print('image.shape is:', image.shape)
    imgShape = image.shape
    for i in range(n):
        x = random.randint(0, imgShape[1] - 1 )
        y = random.randint(0, imgShape[0] - 1 )
        if imgShape[2] == 1:
            point = image[y, x]
            point = 255
        elif imgShape[2] == 3:
            image[y, x] = [random.randint(0, 255),
                  random.randint(0, 255), random.randint(0, 255)]

def main( argv = None ):
    if argv is None:
        argv = sys.argv

    image = cv2.imread("../images/pic1.jpg", cv2.IMREAD_COLOR)
    image2 = np.copy(image)

    dir(image)

    print(type(image))
    print("showing image")
    salt(image, 10000)
    ava.cv.utl.show_image_wait_2(image)

    # image = cv.LoadImageM("../images/pic1.jpg", 1)
    image = image2
    b, g, r = image[1, 1]
    print('colorReduce2 started.')
    image3 = ava.cv.utl.color_reduce_2(image, 64)
    print('colorReduce2 finished.')
    print('colorReduce2Vector started.')
    image3 = ava.cv.utl.color_reduce_2(image, 64)
    print('colorReduce2Vector finished.')
    print('image pix:')
    print(image[0, 0, 0])
    print(type(image[0, 0, 0]))
    print('image3 pix:')
    print(image3[0, 0, 0])
    print(type(image3[0, 0, 0]))
    ava.cv.utl.show_image_wait_2(image3)
    ava.cv.utl.show_image_wait_2(image)

    image = cv.LoadImageM("../images/pic1.jpg", 1)


if __name__ == "__main__":
    main()
