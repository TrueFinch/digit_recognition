from tkinter import *
from typing_extensions import Final


class MainWindow(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title(self.__TITLE__)
        self.pack(fill=BOTH, expand=1)
        train_btn = Button(self, text=self.__TRAIN_BTN_TEXT__)

    __TITLE__: Final = "Digit recognition"
    __TRAIN_BTN_TEXT__: Final = "Train"
