#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code testing the segmenting images using watersheds.

This code testing the segmenting images using watersheds.

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


class WatershedSegmenter(object):
    def __init__(self):
        super(WatershedSegmenter, self).__init__()
        self._markers = None

    @property
    def markers(self):
        return self._markers

    @markers.setter
    def markers(self, markers_image):
        self._markers = np.int32(markers_image )

    def process(self, img):
        "img; np.array, -> return markers as np.array"
        # marker needs to be 32s depth
        cv2.watershed(img, self._markers)
        # after watershed, the boundry is -1
        return self.markers

    def get_segmentation(self):
        result = np.uint8(self.markers)
        return result

    def get_watersheds(self):
        # after watershed, the boundry is -1
        result = np.copy(self.markers)
        result[result>=0] = 255
        result[result==-1] = 0
        return np.uint8(result)


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv
    # logger
    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    img_root_path = '../images'
    img_file = os.path.join(img_root_path, 'group.jpg')
    img_file_binary = os.path.join(img_root_path, 'binary.bmp')

    img = cv2.imread(img_file, cv2.CV_LOAD_IMAGE_COLOR)
    img_binary = cv2.imread(img_file_binary, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # ava.cv.utl.show_image_wait_2(img) # -------------
    # ava.cv.utl.show_image_wait_2(img_binary) # -------------
    # print('img_binary_32')
    # img_binary_32 = np.int32(img_binary)
    # ava.cv.utl.show_image_wait_2(img_binary_32) # -------------

    # 4 steps to get the markers
    # 1. To get foreground,  erosion 6 time, eliminate noise
    # kernal = None, default 3x3
    img_fg = cv2.erode(img_binary, None, iterations=6)
    # ava.cv.utl.show_image_wait_2(img_fg) # -------------

    # 2. To get background, Dilate 6 time the binary image
    img_bg = cv2.dilate(img_binary, None, iterations=6)
    #ava.cv.utl.show_image_wait_2(img_bg) # --------------

    # 3. bg invert
    # (img, threshold, max value, process_type)
    ret, img_bg = cv2.threshold(img_bg, 1, 128, cv2.THRESH_BINARY_INV)
    #ava.cv.utl.show_image_wait_2(img_bg) # --------------

    # 4. create the Marker image
    img_markers = img_fg + img_bg
    print('img_marker dtype:', img_markers.dtype)
    img_markers = np.uint8(img_markers)
    #ava.cv.utl.show_image_wait_2(img_markers) # --------------

    # create segmenter,
    segmenter = WatershedSegmenter()
    segmenter.markers = img_markers

    segmenter.process(img)
    #print("markers:")
    img_markers = segmenter.markers
    #ava.cv.utl.show_image_wait_2(img_markers) #----------------

    #print("segmentation:")
    img_seg = segmenter.get_segmentation()
    #ava.cv.utl.show_image_wait_2(img_seg) #----------------

    #print("watersheds:")
    img_watersheds = segmenter.get_watersheds()
    #ava.cv.utl.show_image_wait_2(img_watersheds) #----------------

    plt.subplot(2, 3, 1), plt.imshow(img)
    plt.title('original')
    plt.subplot(2, 3, 3), plt.imshow(img_binary, 'gray')
    plt.title('binary')
    plt.subplot(2, 3, 4), plt.imshow(img_markers, 'gray')
    plt.title('markers')
    plt.subplot(2, 3, 5), plt.imshow(img_seg, 'gray')
    plt.title('segmentation')
    plt.subplot(2, 3, 6), plt.imshow(img_watersheds, 'gray')
    plt.title('watersheds')

    plt.show()

    exit() # ===================


if __name__ == "__main__":
    main()
