#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv

from os.path import exists

script, fromFile, toFile = argv

print ("Copying from %r to %r" % (fromFile, toFile))
fromFileHandle = open(fromFile)
fromFileContent = fromFileHandle.read()

print("Target file exist? %r" % exists(toFile))
print("ctl-c to about, enter to continue")
input()

toFileHandle = open(toFile, "w")
toFileHandle.write(fromFileContent)

print("All done")
fromFileHandle.close()
toFileHandle.close()
