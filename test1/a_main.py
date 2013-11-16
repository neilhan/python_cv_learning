#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, uncode_literals

import sys
import getopt

class Usage(Exception):
	def __init__(sely, msg):
		self.msg = msg

def main(argv=None):
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt(argv[1:], "h", ["help"])
		except getopt.error, msg:
			raise Usage(msg)
	except Usage, err:
		print(err.msg, file=sys.stderr)
		print("for help use --help", file=sys.stderr)
		return 2

if __name__ == "__main__":
	sys.exit(main())
