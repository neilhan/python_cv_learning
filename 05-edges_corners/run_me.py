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

    def get_edges(self, img):
        """ img: np.array, -> np.array

        This function will find edges.
        """
        # creat an rectangle for edge finding
        # size(3, 3), anchor(1, 1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3), (1, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3), (1, 1))
        img_ret = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
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
        img_ret = cv2.dilate(img, kernal_cross)
        ava.cv.utl.show_image_wait_2(img_ret) # ------------------

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
    img_file = os.path.join(img_root_path, 'binary.bmp')

    img = cv2.imread(img_file)

    # get the Morphology
    morph = Morphology()
    # find edge
    img_edges = morph.get_edges(img)
    # by default 3x3 element is used
    ava.cv.utl.show_image_wait_2(img_edges) # -------------

    # get the corners
    img_corners = morph.get_corners(img)

    exit() # ===================


if __name__ == "__main__":
    main()
