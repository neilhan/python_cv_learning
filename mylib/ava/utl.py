#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import logging
import logging.config
import json

import cv2
import numpy as np


def time_this(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        argsStr = unicode(args)[:10]
        kwStr = unicode(kw)[:10]
        print("%r (%10r, %10r) %2.2f sec" % \
            (method.__name__, argsStr, kwStr, te - ts))
        return result

    return timed


def dir_methods_help(obj, spacing=10, collapse=1):
    """ Prints methods and doc strings.
    Takse modlue, class, list, dictionary, or string."""
    methodList = [m for m in dir(obj) if callable(getattr(obj, m))]
    strProc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (m, strProc(str(getattr(obj, m).__doc__)))
                     for m in methodList]
                    ))

def polyline(img, points, color):
    shape = points.shape
    if len(shape) != 2 and (len(shape) > 1 and shape[1] != 2):
        print('incoming points needs to be np array, in shape (-1,2)')
        return
    xp = None
    yp = None
    for x, y in points:
        if xp is None:
            xp = x
            yp = y
        else:
            cv2.line(img,(xp, yp), (x, y), color)
            xp = x
            yp = y
        print(x, y)


def setup_logging( \
    default_path='../log_config.json', \
    default_level=logging.INFO, \
    env_key='LOG_CFG' \
):
    config_file = default_path
    if os.path.exists(config_file):
        with file(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


