from tkinter import *
from tkinter.ttk import *


class MainWindow(Frame):
    def __init__(self, parent):
        # Frame.__init__(self, parent, background="white")
        s = Style()
        s.configure('My.TFrame', background='white')
        super().__init__(parent, style='My.TFrame')
        self.parent = parent
        self.__initUI__()

    def __initUI__(self):
        self.parent.title(self.__TITLE__)
        self.pack(fill=BOTH, expand=1)

        frame_top = Frame(self)
        frame_top.pack(fill=X)

        self.train_btn = Button(frame_top, text=self.__TRAIN_BTN_TEXT__)
        self.train_btn.pack(side=LEFT, padx=5, pady=5)

        self.predict_btn = Button(frame_top, text=self.__PREDICT_BTN_TEXT__)
        self.predict_btn.pack(side=LEFT, padx=5, pady=5)

        self.frame_bottom = Frame(self)

        self.train_pb = Progressbar(self.frame_bottom, orient=HORIZONTAL, length=100, mode='determinate')
        self.train_pb.pack(side=LEFT, padx=5, pady=5)

    def set_train_callback(self, fun):
        self.train_btn.bind('<Button-1>', fun)

    def set_predict_callback(self, fun):
        self.predict_btn.bind('<Button-1>', fun)

    def show_train_pg(self, flag: bool):
        if flag:
            self.frame_bottom.pack(side=BOTTOM, fill=X)
        else:
            self.pack_forget()

    __TITLE__ = "Digit recognition"
    __TRAIN_BTN_TEXT__ = "Train"
    __PREDICT_BTN_TEXT__ = "Predict"
