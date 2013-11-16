#!/usr/bin/env python
# -*- coding: utf-8 -*-

def print3(*args):
	arg1, arg2, arg3 = args
	print("(1)arg1:%r arg2:%r" % (arg1, arg2))

def print2(arg1, arg2):
	print("(2)arg1:%r arg2:%r" % (arg1, arg2))

def print1(arg1):
	print("arg1:%r" % arg1)

def print0():
	print("hi, all good. ")

print3("111", "2222", "333")
print2("123", "2234")
print1("only you")
print0()
