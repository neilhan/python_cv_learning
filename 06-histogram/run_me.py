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


class Histogram1D:
    def __init__(self):
        print("Histogram1D created.")

    def getHistogram(self, image):
        # how many image, channels, mask, result, ?D histogram,
        # number of bins, pixel value range
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist


@ava.utl.time_this
def main(argv=None):
    if argv is None:
        argv = sys.argv

    ava.utl.setup_logging()
    logger = logging.getLogger('RunMe.main')

    logger.debug('starting main.')

    # histogram helper creating
    histogram = Histogram1D()

    image1 = cv2.imread("../pic1.jpg")
    # testing sparseMat
    # sparse_hist = sp.csr_matrix((256,1), dtype=np.int32)
    # cv2.calcHist([image1],[0], None, [256], [0, 256], sparse_hist)
    # print(sparse_hist.shape)
    logger.debug(type(image1))
    logger.debug('-------------------------------------')
    logger.debug("image.shape: " + unicode(image1.shape))
    logger.debug('image: ' + unicode(image1))
    logger.debug('image dtype: ' + unicode(image1.dtype))
    cv2.waitKey(10000)

    # split image
    image1BW = np.zeros((image1.shape[0], image1.shape[1]), dtype=image1.dtype)
    # b = cv2.split(a)[0]
    image1BW[:, :] = image1[:, :, 0]
    logger.debug("type(image1BW): " + str(type(image1BW)))
    logger.debug("image1BW.shape: " + str(image1BW.shape))
    logger.info("Displaying the B channel::::::: ")
    ava.cv.utl.show_image_wait_2(image1BW)

    # apply look up table
    lut = np.arange(255, -1, -1, dtype=image1.dtype)
    lut[0:60] = 255
    lut[200:] = 0

    image1BWR = cv2.LUT(image1BW, lut)
    print("lut shape:", lut.shape)
    print("lut dtype:", lut.dtype)
    print('lut:', lut)
    logger.info("Displaying the cv2.LUT()::::::: ")
    ava.cv.utl.show_image_wait_2(image1BWR)

    # lut.shape = (256, 1)
    lut3 = np.column_stack((lut, lut, lut))
    lutIdxDot = np.array([0, 1, 2], dtype=int)
    lutIdx0 = np.zeros(image1.shape[0] * image1.shape[1], dtype=int)
    lutIdx1 = np.ones(image1.shape[0] * image1.shape[1], dtype=int)
    lutIdx2 = lutIdx1 * 2
    lutIdx = np.column_stack((lutIdx0, lutIdx1, lutIdx2))
    lutIdx.shape = image1.shape
    #print("lut dtype:", lut.dtype)
    #print(lut)
    image1Rev = lut3[image1, lutIdx]
    # image1Rev = lut[image1]
    ava.cv.utl.show_image_wait_2(image1Rev)

    hist = histogram.getHistogram(image1)
    print("type of hist is:", type(hist))
    print("hist.shape: ", hist.shape)
    plt.plot(hist)
    plt.show()

    logger.info('cv2.normalize() :::::::::::::::::')
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    plt.plot(hist)
    plt.show()

    # for i, histVal in enumerate(hist):
    #     print("hist(", i, "):", histVal)
    #     image1[int(histVal), i] = (0, 0, 0)
    ava.cv.utl.show_image_wait_2(image1)

if __name__ == "__main__":
    main()
