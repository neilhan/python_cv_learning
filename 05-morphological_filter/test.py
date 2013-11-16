#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code compared a couple histograms.

"""

from __future__ import absolute_import, division, \
    print_function, unicode_literals

# import ipdb; ipdb.set_trace() ; # debugging-------
import sys
import logging
import os

import numpy as np
import scipy.sparse as sp
import cv2
cv2.namedWindow('GetArroundASegmentationFailure', 0)
cv2.destroyWindow('GetArroundASegmentationFailure')
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl

import cv2
import numpy as np

base = cv2.imread('../images/waves.jpg')
test1 = cv2.imread('../images/beach.jpg')

rows,cols = base.shape[:2]

basehsv = cv2.cvtColor(base,cv2.COLOR_BGR2HSV)
test1hsv = cv2.cvtColor(test1,cv2.COLOR_BGR2HSV)

halfhsv = basehsv[rows/2:rows-1,cols/2:cols-1].copy()  # Take lower half of the base image for testing

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange                                  # ranges = [0,180,0,256]


histbase = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
cv2.normalize(histbase,histbase,0,255,cv2.NORM_MINMAX)

histtest1 = cv2.calcHist(test1hsv,[0,1],None,[180,256],ranges)
cv2.normalize(histtest1,histtest1,0,255,cv2.NORM_MINMAX)

base_base = cv2.compareHist(histbase, histbase, 0)
base_test1 = cv2.compareHist(histbase, histtest1, 0)
print("Method: {0} -- base-half: {1}".format(base_base,base_test1))

