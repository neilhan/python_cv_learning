#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import cv
import ava.cv.utl

image = cv.LoadImageM("../images/pic1.jpg", 1)

print(type(image))
dst = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 3)
laplace = cv.Laplace(image, dst)
print(type(laplace))

print("showing image")
ava.cv.utl.show_image_wait(image)
print("showing laplace")
ava.cv.utl.show_image_wait(dst)
