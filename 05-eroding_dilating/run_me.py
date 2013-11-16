#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code tests the eroding and dilating.

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

    # by default 3x3 element is used
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_eroded = cv2.erode(img, kernel)

    ava.cv.utl.show_image_wait_2(img_eroded) # -------------

    img_dilated = cv2.dilate(img_eroded, kernel)
    ava.cv.utl.show_image_wait_2(img_dilated) # -------------

    exit() # ===================


if __name__ == "__main__":
    main()
