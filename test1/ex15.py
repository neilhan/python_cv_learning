#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv

script, fileName = argv

fileHandle = open(fileName)

print("file:", fileName)
fileContent = fileHandle.read()
print(fileContent)
fileContent2 = fileHandle.read()

print("Content read 2: %r" % fileContent2)
