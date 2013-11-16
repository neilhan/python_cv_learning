#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ubuntu default opencv has a bug. cv2.polylines will cause an error.

"""

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import cv2
import numpy as np


image = np.zeros((768, 1024, 3), dtype='uint8')

points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])
cv2.polylines(image, [points], 1, (255, 255, 0))

winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, image)
cv2.waitKey()
cv2.destroyWindow(winname)
