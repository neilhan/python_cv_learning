#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

import numpy as np
import cv2

import ava.utl

h = np.zeros((300, 256, 3))
c = (255,0,0)
ava.utl.polyline(h, np.array([[10,10], [20,20]]), c)
