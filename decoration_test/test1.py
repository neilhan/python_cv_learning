#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, \
    print_function, unicode_literals

def mod(method):
	method.__name__ = "heh"
	return method

@mod
def modFunction():
	pass

print(modFunction.__name__)
