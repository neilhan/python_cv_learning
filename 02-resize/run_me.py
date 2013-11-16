#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import cv

original = cv.LoadImageM("../images/pic1.jpg")
print(type(original))

thumbnail = cv.CreateMat(original.rows // 10, original.cols // 10, cv.CV_8UC3)
cv.Resize(original, thumbnail)

cv.NamedWindow("The window", cv.CV_WINDOW_AUTOSIZE)
image = thumbnail

cv.ShowImage("The window", image)
cv.WaitKey(10000)

original = cv.LoadImage("../images/pic1.jpg")
print(type(original))
