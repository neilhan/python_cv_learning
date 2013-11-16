#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import cv2
import ava.cv.utl

# grayscale = 0
image = cv2.imread("../images/pic1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)

print(image.dtype)
img_laplace = cv2.Laplacian(image, cv2.CV_8U, ksize=3)

ava.cv.utl.show_image_wait_2(image)
ava.cv.utl.show_image_wait_2(img_laplace)
