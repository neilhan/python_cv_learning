#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, \
    print_function, unicode_literals

# import ipdb; ipdb.set_trace() ; # debugging--------------------------------------
import sys
import logging

import numpy as np
import scipy.sparse as sp
import cv2
cv2.namedWindow('GetArroundASegmentationFailure', 0)
cv2.destroyWindow('GetArroundASegmentationFailure')
import matplotlib.pyplot as plt

import ava.utl
import ava.cv.utl


def get_hue_histogram(img_hsv, min_saturation=0):
    """ img: np.array, min_saturation int, -> historgram as np.array """
    assert img_hsv != None, "img can't be null"
    assert img_hsv.shape[2] == 3, 'Expecting 3 channel image'

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    h = h[..., np.newaxis]
    s = s[..., np.newaxis]

    if min_saturation > 0:
        ret, saturation_mask = cv2.threshold(s, min_saturation, 255, cv2.THRESH_BINARY)

    print('saturation_mask: ==================')
    ava.cv.utl.show_image_wait_2(saturation_mask) # ------------
    masked_img = cv2.bitwise_and(img_hsv, cv2.merge([saturation_mask, saturation_mask, saturation_mask]))
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)
    ava.cv.utl.show_image_wait_2(masked_img) # ------------
    hist = cv2.calcHist([img_hsv], [0], saturation_mask, [180], [0, 180])
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


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv

    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    img_path = '../images/baboon1.jpg'
    img_2_path = '../images/baboon3.jpg'

    # read image
    src_img_gray = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    src_img_bgr = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
    src_img_2_bgr = cv2.imread(img_2_path, cv2.CV_LOAD_IMAGE_COLOR)

    src_img_hsv = cv2.cvtColor(src_img_bgr, cv2.COLOR_BGR2HSV)
    src_img_2_hsv = cv2.cvtColor(src_img_2_bgr, cv2.COLOR_BGR2HSV)
    src_img_cp = src_img_bgr
    src_img_2_cp = src_img_2_bgr
    ava.cv.utl.show_image_wait_2(src_img_bgr) # ---------

    # tracking window
    x, y, w, h = (110, 260, 35, 40)
    track_window = (x, y, w, h)
    # draw rectangle
    # (x1, y1), (x2, y2), color
    cv2.rectangle(src_img_bgr, (110, 260), (110 + 35, 260 + 40), (255,200,100))
    ava.cv.utl.show_image_wait_2(src_img_bgr) # --------

    # get the hist_sample
    hist_sample = get_hue_histogram(src_img_hsv[260:(260 + 40), 110:(110 + 35)], 65)

    # find in the image_2
    src_img_2_masked = find_content(src_img_2_hsv, hist_sample)
    # termination criteria
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    ret, track_window = cv2.meanShift(src_img_2_masked, track_window, term_crit)
    x, y, w, h = track_window
    print('found window: ', track_window)
    cv2.rectangle(src_img_2_cp, (x, y), (x + w, y + h), (255, 255, 0))

    # result = np.vstack([src_img_2_cp, src_img_2_masked_bgr])
    ava.cv.utl.show_image_wait_2(src_img_2_cp, 0) # --------------------

    cv2.destroyAllWindows()

    exit()


if __name__ == "__main__":
    main()
