from tkinter import *
from tkinter.ttk import *


class MainWindow(Frame):
    def __init__(self, parent):
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

        self.train_btn = Button(frame_top, text=self.__BTN_TEXT_TRAIN__)
        self.train_btn.pack(side=LEFT, padx=5, pady=5)

        self.predict_btn = Button(frame_top, text=self.__BTN_TEXT_PREDICT__)
        self.predict_btn.pack(side=LEFT, padx=5, pady=5)

        self.frame_bottom = Frame(self)

        self.label_epoch = Label(self.frame_bottom)
        self.label_epoch.pack(side=LEFT, padx=5, pady=5)

        self.train_pb = Progressbar(
            self.frame_bottom, orient=HORIZONTAL,
            length=self.progress_bar_length,
            mode='determinate'
        )
        self.train_pb.config(mode="determinate", maximum=100, value=0)
        self.train_pb.pack(side=LEFT, padx=5, pady=5)

    def set_train_btn_listener(self, fun):
        self.train_btn.bind('<Button-1>', fun)

    def set_predict_callback(self, fun):
        self.predict_btn.bind('<Button-1>', fun)

    def show_train_pg(self, flag: bool):
        if flag:
            self.frame_bottom.pack(side=BOTTOM, fill=X)
        else:
            self.frame_bottom.pack_forget()
        self.frame_bottom.update()

    def increase_pg(self, value):
        self.set_pd(value)

    def increase_epoch(self):
        if not self.is_first_epoch:
            self.set_epoch(self.epoch_count + 1)
        else:
            self.is_first_epoch = False
        self.set_pd(0)

    def set_epoch(self, value: int):
        self.epoch_count = value
        self.label_epoch.config(text=self.__TEXT_LABEL_EPOCH__ % value)
        self.label_epoch.update()

    def reload_pb(self):
        self.set_epoch(1)
        self.is_first_epoch = True
        self.set_pd(0)
        self.progress = 0

    def set_pd(self, value):
        self.progress += value
        self.train_pb.step(value)
        self.train_pb.update()

    is_first_epoch = True
    epoch_count = 1
    progress = 0
    progress_bar_length = 100
    __TITLE__ = "Digit recognition"
    __BTN_TEXT_TRAIN__ = "Train"
    __BTN_TEXT_PREDICT__ = "Predict"
    __TEXT_LABEL_EPOCH__ = "Epoch №%d"
