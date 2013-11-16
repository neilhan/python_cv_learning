#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv

script, inFile = argv

def printAll(file):
	print(file.read())

def printALine(lineNum, file):
	print(lineNum, file.readline())

def rewind(file):
	file.seek(0)

inFileHandle = open(inFile)

printAll(inFileHandle)

rewind(inFileHandle)

lineNum = 1
printALine(lineNum, inFileHandle)
lineNum = lineNum + 1
printALine(lineNum, inFileHandle)
lineNum = lineNum + 1
printALine(lineNum, inFileHandle)
