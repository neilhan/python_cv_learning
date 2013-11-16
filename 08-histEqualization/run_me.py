#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Doing a histogram equalization on an image.

"""

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


def main(argv=None):
    if argv == None:
        argv = sys.argv

    image1 = cv2.imread("../images/pic2.jpg", 1)
    print(type(image1))
    print("image.shape:", image1.shape)
    print('image:', image1)
    print('image dtype:', image1.dtype)

    # split image
    image1BW = np.zeros((image1.shape[0], image1.shape[1]), dtype=image1.dtype)
    # b = cv2.split(a)[0]
    image1BW[:,:] = image1[:,:, 0]
    print("type(image1BW)", type(image1BW))
    print("image1BW.shape:", image1BW.shape)
    ava.cv.utl.show_image_wait_2(image1BW)

    hist, bins = np.histogram(image1BW.flatten(), 256, [0, 256])

    cdf = hist.cumsum() # cumulative sum
    cdf_normalized = cdf * hist.max() / cdf.max() # not necessary

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(image1BW.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()

    # cdf_m = np.ma.masked_equal(cdf, 400)
    cdf_m = np.ma.masked_less(cdf, 400)
    print('cdf_m = np.ma.masted_less():', cdf_m)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / ( cdf_m.max() - cdf_m.min() )
    print('cdf_m = (cdf_m - cdf_m.min()) * 255 / ( cdf_m.max() - cdf_m.min() ):', cdf_m)
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    print("cdf = np.ma.filled(cdf_m, 0).astype('uint8'):", cdf)

    image2BW = cdf[image1BW]
    ava.cv.utl.show_image_wait_2(image2BW)
    print('image2BW is displayed')

    # checking the new plot
    hist = cv2.calcHist([image2BW], [0], None, [256], [0, 256])
    cdf2 = hist.cumsum()
    cdf2_norm = cdf2 * hist.max() / cdf2.max()

    # cv2 has equalizeDist(image)
    image3 = cv2.equalizeHist(image1BW)
    twoImages = np.hstack((image1BW, image3))
    ava.cv.utl.show_image_wait_2(twoImages)

    plt.plot(cdf2_norm, color = 'b')
    plt.hist(image2BW.flatten(), 256, [0, 256], color = 'g')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()

    # apply look up table, reverse testing.
    lut = np.arange(255, -1, -1, dtype = image1.dtype )
    image1BWR = cv2.LUT(image1BW, lut)
    print("lut shape:", lut.shape)
    print("lut dtype:", lut.dtype)
    print('lut:', lut)
    ava.cv.utl.show_image_wait_2(image1BWR)

    # lut.shape = (256, 1)
    lut = np.column_stack((lut, lut, lut))

    #print("lut shape:", lut.shape)
    #print("lut dtype:", lut.dtype)
    #print(lut)
    # image1Rev = cv2.LUT(image1, lut)
    lutIdx0 = np.zeros(image1.shape[0] * image1.shape[1], dtype=int)
    lutIdx1 = np.ones(image1.shape[0] * image1.shape[1], dtype=int)
    lutIdx2 = lutIdx1 * 2
    lutIdx = np.column_stack((lutIdx0, lutIdx1, lutIdx2))
    lutIdx.shape = image1.shape
    image1Rev = lut[image1, lutIdx]
    ava.cv.utl.show_image_wait_2(image1Rev)

    hist = histogram.getHistogram(image1)
    print("type of hist is:", type(hist))
    print("hist.shape: ", hist.shape)
    plt.plot(hist)
    plt.show()

    # for i, histVal in enumerate(hist):
    #    print("hist(", i, "):", histVal)
    cv2.normalize( hist, hist, 0, 255, cv2.NORM_MINMAX )
    for i, histVal in enumerate(hist):
        print("hist(", i, "):", histVal)
        image1[int(histVal), i] = (0, 0, 0)
    ava.cv.utl.show_image_wait_2(image1)


if __name__ == "__main__":
    main()
