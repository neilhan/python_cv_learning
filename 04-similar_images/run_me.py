#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program does an image search.

Image search. Use histogram similarity functions.

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


def get_hue_histogram(img_hsv, min_saturation=0):
    """ img: np.array, min_saturation int, -> historgram as np.array. """
    assert img_hsv is not None, "img can't be null"
    assert img_hsv.shape[2] == 3, 'Expecting 3 channel image'

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    h = h[..., np.newaxis]
    s = s[..., np.newaxis]

    saturation_mask = None

    if min_saturation > 0:
        ret, saturation_mask = cv2.threshold(s, min_saturation, 255, cv2.THRESH_BINARY)

    hist = cv2.calcHist([img_hsv], [0], saturation_mask, [180], [0, 180])
    return hist


def get_hs_histogram(img_hsv, min_saturation=0):
    """ img: np.array, min_saturation int, -> historgram as np.array. """
    assert img_hsv is not None, "img can't be null"
    assert img_hsv.shape[2] == 3, 'Expecting 3 channel image'

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    h = h[..., np.newaxis]
    s = s[..., np.newaxis]

    saturation_mask = None

    if min_saturation > 0:
        ret, saturation_mask = cv2.threshold(s, min_saturation, 255, cv2.THRESH_BINARY)

    hist = cv2.calcHist([img_hsv], [0,1], saturation_mask, [180, 256], [0, 180, 0, 256])
    return hist


def find_content(img_hsv, hist_sample):
    """ img hsv, hist_sample as np.array, -> 1 channel distance """
    src_img_cp = img_hsv
    # normalize the sample histogram
    cv2.normalize(hist_sample, hist_sample, 0, 179, cv2.NORM_MINMAX)
    distance = cv2.calcBackProject([img_hsv], [0], hist_sample, [0, 180], 0.5)

    print('ssssssssssssssssssssss distance -------------------')
    # show the distance
    ava.cv.utl.show_image_wait_2(distance) # ------------

    # convolute with circular, morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(distance, -1, kernel, distance)

    print('==================== distance convoluted -------------------')
    # show the smoothed distance
    ava.cv.utl.show_image_wait_2(distance) # ------------

    # threshold
    ret, thresh = cv2.threshold(distance, 55, 180, cv2.THRESH_BINARY)
    # thresh = cv2.merge([thresh, thresh, thresh])

    # do the bitwise_and
    #result = cv2.bitwise_and(src_img_cp, thresh)
    return thresh


class ImageComparator(object):
    def __init__(self):
        super(ImageComparator, self).__init__()
        self._ref_img = None
        self._ref_img_color_reduced = None
        self._ref_img_histogram = None
        self._color_reduction_factor = 32

    @property
    def ref_img(self):
        return self._ref_img

    @ref_img.setter
    def ref_img(self, img):
        self._ref_img = img
        self._ref_img_color_reduced = \
            ava.cv.utl.color_reduce_2(img, self._color_reduction_factor)
        ref_img_hsv = cv2.cvtColor(self._ref_img_color_reduced, cv2.COLOR_BGR2HSV)
        self._ref_img_histogram = \
            get_hs_histogram(ref_img_hsv)

    @ref_img.deleter
    def ref_img(self):
        del(self._ref_img)

    @property
    def ref_img_color_reduced(self):
        return self._ref_img_color_reduced

    def compare(self, the_img):
        the_img_color_reduced = \
            ava.cv.utl.color_reduce_2(the_img, self._color_reduction_factor)
        the_img_hsv = cv2.cvtColor(the_img_color_reduced, cv2.COLOR_BGR2HSV)
        the_img_hist = get_hs_histogram(the_img_hsv)

        # print('h1.type():', self._ref_img_histogram.type())
        # print('h2.type():', the_img_hist.type())
        score = cv2.compareHist(self._ref_img_histogram, the_img_hist, 0)
        return score


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv
    # logger
    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    img_root_path = '../images'
    img_files = [
        os.path.join(img_root_path, 'waves.jpg'),
        os.path.join(img_root_path, 'beach.jpg'),
        os.path.join(img_root_path, 'dog.jpg'),
        os.path.join(img_root_path, 'polar.jpg'),
        os.path.join(img_root_path, 'bear.jpg'),
        os.path.join(img_root_path, 'lake.jpg'),
        os.path.join(img_root_path, 'moose.jpg') ]

    img_comparator = ImageComparator()
    img_comparator.ref_img = cv2.imread(
        os.path.join(img_root_path, 'waves.jpg'))

    # compare
    for img_file in img_files:
        the_img = cv2.imread(img_file)
        img_size = the_img.shape[0] * the_img.shape[1]
        score = img_comparator.compare(the_img)
        print(img_file + ', score: %6.4f' % (score))
    ava.cv.utl.show_image_wait_2(img_comparator.ref_img_color_reduced)

    exit() # ===================


if __name__ == "__main__":
    main()
