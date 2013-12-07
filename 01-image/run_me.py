#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing read image, wait, show image, save image.

Author: Neil Han

"""


from __future__ import absolute_import, division, \
    print_function, unicode_literals

import cv2
cv2.namedWindow('WorkAroundTheCoreDump')
cv2.destroyWindow('WorkAroundTheCoreDump')
import numpy as np


cv2.namedWindow("The window", cv2.CV_WINDOW_AUTOSIZE)
image = cv2.imread("../images/pic1.jpg")

x = 20
y = 550

cv2.putText(image, "Hi, pibbles, snow. ", (x, y),
            cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=255)

cv2.imshow("The window", image)
cv2.waitKey(10 * 1000)

# flip an image
image = cv2.imread('../images/pic1.jpg', cv2.CV_LOAD_IMAGE_COLOR)
image = cv2.flip(image, 0)
cv2.imshow('Flipped', image)
cv2.waitKey(0)

# gray image
image = cv2.imread('../images/pic1.jpg', cv2.CV_LOAD_IMAGE_COLOR)
imgGray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
cv2.imshow('colorhist', imgGray)
cv2.waitKey(0)

#cv.SaveImage("../pic1-copy.jpg", image)
cv2.imwrite('./pic1_copy.jpg', image)
