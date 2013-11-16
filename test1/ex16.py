#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv

script, fileName = argv

print ("Erasing %r" % fileName)
print("ctl-c to cancel, or enter to continue")
input(">")

target = open(fileName, "w")

print("cleaning / truncating file. Goodbye")
target.truncate()

target.write("line1")
target.write("""line2
line3""")

print("close.")
target.close()
