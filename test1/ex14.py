#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, uncode_literals

from sys import argv

script, user_name = argv

prompt = "> "

print("Hi, %s, I'm the %s script." % (user_name, script))
print ("I'd like to ast you some questions.")
print ("Are you really %s?" % user_name)
tf = input(prompt)

print("where do you live %s?" % user_name)
location = input(prompt)


print ("""
Alright, said %r are you.
Live in %r.
Nice. """ % (tf, location))
