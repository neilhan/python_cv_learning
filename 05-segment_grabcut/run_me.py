#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code testing the segmenting images using GrabCut method.

This code testing the segmenting images using GrabCut method.
Author: Neil.Han@gmail.com

! This sample code is a copy from https://github.com/Itseez/opencv/blob/master/samples/python2/grabcut.py
@ 2013-11-29
The original code was created by abidrahmank, and SpecLad

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


# this is the model object, carring the state, image etc
model = None


class GrabcutModel(object):
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]
    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]

    DRAW_BG = {'color': BLACK, 'val': 0}
    DRAW_FG = {'color': WHITE, 'val': 1}
    DRAW_PR_BG = {'color': RED, 'val': 2}
    DRAW_PR_FG = {'color': GREEN, 'val': 3}

    def reset(self):
        self.rectangle = False
        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rect_over = False
        self.rect_or_mask = 100
        self.value = GrabcutModel.DRAW_FG  # drawing brush init to FG
        self.thickness = 3
        self.img = self.img_backup.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, dtype=np.uint8)
        self.ix = 0
        self.iy = 0

    def __init__(self, img):
        super(GrabcutModel, self).__init__()
        self.img_backup = img
        # self.reset()
        self.rectangle = False
        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rect_over = False
        self.rect_or_mask = 100
        self.value = GrabcutModel.DRAW_FG  # drawing brush init to FG
        self.thickness = 3
        self.img = self.img_backup.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.img.shape, dtype=np.uint8)
        self.ix = 0
        self.iy = 0


    def get_output_img(self):
        bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
        result = np.hstack((self.img_backup, bar, self.img, bar, self.img_result))
        return result


def onmouse(event, x, y, flags, param):
    global model

    # Draw rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        model.rectangle = True
        model.ix, model.iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # draw a new rectangle
        if model.rectangle == True:
            model.img = model.img_backup.copy()
            cv2.rectangle(model.img, (model.ix, model.iy), (x, y), model.BLUE, 2)
            model.rect_or_mask = 0
    elif event == cv2.EVENT_RBUTTONUP:
        model.rectangle = False
        model.rect_over = True
        cv2.rectangle(model.img, (model.ix, model.iy), (x, y), model.BLUE, 2)
        model.rect = (model.ix, model.iy, abs(model.ix - x), abs(model.iy - y))
        model.rect_or_mask = 0
        print('Press "n" a few times until no further change.')

    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        if model.rect_over == False:
            print('Please use your mouse right button to draw a rectangle.')
        else:
            model.drawing = True
            # draw a dot
            cv2.circle(model.img, (x, y), model.thickness, model.value['color'], -1)
            cv2.circle(model.mask, (x, y), model.thickness, model.value['val'], -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if model.drawing == True:
            # draw a dot
            cv2.circle(model.img, (x, y), model.thickness, model.value['color'], -1)
            cv2.circle(model.mask, (x, y), model.thickness, model.value['val'], -1)
    elif event == cv2.EVENT_LBUTTONUP:
        if model.drawing == True:
            model.drawing = False
            # draw a dot
            cv2.circle(model.img, (x, y), model.thickness, model.value['color'], -1)
            cv2.circle(model.mask, (x, y), model.thickness, model.value['val'], -1)
    # /// end onmouse


@ava.utl.time_this
def main(argv=None):
    global model

    if argv is None:
        argv = sys.argv
    # logger
    ava.utl.setup_logging()
    logger = logging.getLogger(__name__).getChild('main')

    logger.debug('starting main.')

    print('''Keys:
    'ESC' to exit
    0 to mark sure background with mouse button
    1 to mark sure backgroud
    2 to do PR_BG drawing, probable background
    3 to do PR_FG drawing, probable foreground
    n to update the segmentation
    r to reset the setup
    s to save the results to ./test_output.png''')

    img_root_path = '../images'

    if len(argv) >= 2:
        img_file = argv[1]  # use provided file
    else:
        print('Use default image.')
        img_file = os.path.join(img_root_path, 'dog2.jpg')  # 'tiger.jpg')

    img = cv2.imread(img_file, cv2.CV_LOAD_IMAGE_COLOR)

    cv2.namedWindow('output')
    cv2.namedWindow('input')
    # cv2.namedWindow('mask')
    # cv2.namedWindow('mask2')
    cv2.setMouseCallback('input', onmouse)
    cv2.moveWindow('input', img.shape[1] + 10, 90)

    model = GrabcutModel(img)  # init the model

    # start the main loop
    while(1):
        cv2.imshow('output', model.output)
        cv2.imshow('input', model.img)
        k = 0xFF & cv2.waitKey(10)

        if k == 27:  # esc exit
            break
        elif k == ord('0'):  # draw with BG
            print('Mark background regions with left mouse button.')
            model.value = GrabcutModel.DRAW_BG
        elif k == ord('1'):  # draw with FG
            print('Mark foreground regions with left mouse button.')
            model.value = GrabcutModel.DRAW_FG
        elif k == ord('2'):  # draw with PR_BG
            print('Mark probable background regions with left mouse button.')
            model.value = GrabcutModel.DRAW_PR_BG
        elif k == ord('3'):  # draw with PR_FG
            print('Mark probable foreground regions with left mouse button.')
            model.value = GrabcutModel.DRAW_PR_FG
        elif k == ord('r'):  # reset model
            print('Rest segments.')
            model.reset()
        elif k == ord('s'):  # save
            result = model.get_output_img()
            cv2.imwrite('grabcut_output.png', result)
        elif k == ord('n'):  # segment the image
            print('For finer touchups, mark foreground and background \
                  after pressing keys 0-3 and press "n"')

            if model.rect_or_mask == 0:  # grabcut with rect
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(model.img_backup, model.mask, model.rect,
                            bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                model.rect_or_mask = 1
            elif model.rect_or_mask == 1:  # grabcut with mask
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                cv2.grabCut(model.img_backup, model.mask, model.rect,
                            bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

        model.mask2 = np.where((model.mask == 1) + (model.mask == 3), 255, 0).astype('uint8')
        # cv2.imshow('mask2', model.mask2)
        # cv2.imshow('mask', model.mask)
        model.output = cv2.bitwise_and(model.img_backup, model.img_backup, mask=model.mask2)
    # /// while loop ends

    cv2.destroyAllWindows()

    exit()  # ===================


if __name__ == "__main__":
    main()
