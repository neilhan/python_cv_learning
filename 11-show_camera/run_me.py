#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Display web camera """

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import sys
import cv2
cv2.namedWindow('WorkAroundTheCoreDump')
cv2.destroyWindow('WorkAroundTheCoreDump')
import numpy as np
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl

def getHistogram(image):
    # how many image, channels, mask, result, ?D histogram,
    # number of bins, pixel value range
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


@ava.utl.time_this
def main(argv=None):
    cam = cv2.VideoCapture()

    if cam is not None:
        cam.open(-1)
        print('Camera is open:', cam.isOpened())
        retval, image = cam.read()
        while retval:
            retval, image = cam.read(image)
            print('image type:', type(image))
            print('retval type:', type(retval))
            cv2.imshow('WorkAroundTheCoreDump', image)
            the_key = cv2.waitKey(1)
            if the_key == ord('q'):
                break
            #    cv2.imshow('hist', vis)

        print('Press any key to terminate this program.')
        cv2.waitKey()
        cam.release()


if __name__ == "__main__":
    main()
