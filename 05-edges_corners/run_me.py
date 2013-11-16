#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code testing the Morphology.

This code can find borders and corners.

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


class Morphology(object):
    def __init__(self):
        super(Morphology, self).__init__()
        self._threshold = -1
        self._kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3), (1, 1))
        self._kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5), (1, 1))
        self._kernel_cross = np.array(
            [[0,0,1,0,0],
             [0,0,1,0,0],
             [1,1,1,1,1],
             [0,0,1,0,0],
             [0,0,1,0,0]], dtype=np.uint8)

        self._kernel_diamond = np.array(
            [[0,0,1,0,0],
             [0,1,1,1,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
             [0,0,1,0,0]], dtype=np.uint8)

        self._kernel_x = np.array(
            [[1,0,0,0,1],
             [0,1,0,1,0],
             [0,0,1,0,0],
             [0,1,0,1,0],
             [1,0,0,0,1]], dtype=np.uint8)


    def get_edges(self, img):
        """ img: np.array, -> np.array

        This function will find edges.
        """
        # creat an rectangle for edge finding
        # size(3, 3), anchor(1, 1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3), (1, 1))
        img_ret = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, self._kernel_3x3)
        img_ret = self.apply_threshold(img_ret)
        return img_ret

    def get_corners(self, img):
        """ img: np.array, -> np.array

        This function will find corners. It does:
            dilate with cross
            erode with diamond
            dilate with X
            erode with square
            Corners are obtained by differentiating the two closed images
        """
        img_1 = cv2.dilate(img, self._kernel_cross)
        print('cross dilate')
        # ava.cv.utl.show_image_wait_2(img_1) # ------------------

        img_1 = cv2.erode(img_1, self._kernel_diamond)
        print('erode diamond')
        # ava.cv.utl.show_image_wait_2(img_1) # ------------------

        img_2 = cv2.dilate(img, self._kernel_x)
        print('x dilate')
        # ava.cv.utl.show_image_wait_2(img_2) # ------------------

        img_2 = cv2.erode(img_2, self._kernel_5x5)
        print('erode square')
        # ava.cv.utl.show_image_wait_2(img_2) # ------------------

        img_1 = cv2.absdiff(img_2,img_1)
        #threshold
        img_1 = self.apply_threshold(img_1)

        return img_1

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self,v):
        self._threshold = v

    @threshold.deleter
    def threshold(self):
        del(self._threshold)

    def apply_threshold(self, img):
        img_ret = img

        if self.threshold > 0:
            ret, img_ret = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY_INV)

        return img_ret


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv
    # logger
    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    img_root_path = '../images'
    img_file = os.path.join(img_root_path, 'building.jpg')

    img = cv2.imread(img_file)

    # get the Morphology
    morph = Morphology()
    morph.threshold = 40
    # find edge
    img_edges = morph.get_edges(img)
    # by default 3x3 element is used
    ava.cv.utl.show_image_wait_2(img_edges) # -------------

    # get the corners
    img_corners = morph.get_corners(img)
    print('Corners')
    ava.cv.utl.show_image_wait_2(img_corners) # -------------

    exit() # ===================


if __name__ == "__main__":
    main()
