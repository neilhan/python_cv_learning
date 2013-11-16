#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing tk window and showing images

Test another line of comments.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import Tkinter
import ava.utl
import ava.cv.utl


class SimpleAppTk(Tkinter.Tk):
    def __init__(self, parent):
        Tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.grid()

        self.inputVar = Tkinter.StringVar()
        self.inputVar.set('Enter here.')
        self.entry = Tkinter.Entry(self, textvariable=self.inputVar)
        self.entry.grid(column=0, row=0, sticky='EW')
        self.entry.bind('<Return>', self.onPressEnter)

        button = Tkinter.Button(self, text='Click here',
                                command=self.onButtonClick)
        button.grid(column=1, row=0)

        self.labelVar = Tkinter.StringVar()
        self.labelVar.set('Hello !')
        label = Tkinter.Label(self, textvariable=self.labelVar,
                              anchor='w', fg='white',
                              bg='blue')
        label.grid(column=0, row=1, columnspan=2, sticky='EW')
        # let resize happen
        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)
        # render
        self.update()
        # set a fixed size
        self.geometry(self.geometry())
        # set the selected text
        self.entry.focus_set()
        self.entry.selection_range(0, Tkinter.END)

    def onButtonClick(self):
        self.labelVar.set(self.inputVar.get() + ' (Clicked)')
        self.entry.selection_range(0, Tkinter.END)
        print('Button in action! clicked!')

    def onPressEnter(self, event):
        self.labelVar.set(self.inputVar.get() + ' (Enter pressed!)')
        self.entry.selection_range(0, Tkinter.END)
        print('Button in action! enter!')


# @ava.utl.time_this
def main(argv=None):
    app = SimpleAppTk(None)
    app.title('SimpleAppTk')
    app.mainloop()


if __name__ == "__main__":
    main()
