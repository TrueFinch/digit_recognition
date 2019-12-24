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
        self.columnconfigure(2, weight=1)

        frame_top = Frame(self, relief=SUNKEN)
        frame_top.grid(row=0, column=0, columnspan=3, sticky=E + W + S + N)

        self.btn_train = Button(frame_top, text=self.__TEXT_BTN_TRAIN__)
        self.btn_train.pack(side=LEFT, padx=5, pady=5)
        self.btn_predict = Button(frame_top, text=self.__TEXT_BTN_PREDICT__)
        self.btn_predict.pack(side=LEFT, padx=5, pady=5)
        self.btn_clear = Button(frame_top, text=self.__TEXT_BTN_CLEAR__)
        self.btn_clear.pack(side=LEFT, padx=5, pady=5)

        frame_canvas = Frame(self, style='My.TFrame')
        frame_canvas.grid(row=1, column=0, columnspan=3, sticky=E + W + S + N)

        self.canvas = Canvas(frame_canvas, bg="white", height=self.canvas_size, width=self.canvas_size)
        self.canvas.pack(side=LEFT, padx=10, pady=10)
        self.btn_clear.bind("<Button-1>", lambda event: self.canvas.delete("all"))

        self.lbl_answer = Label(frame_canvas, background="white", justify=CENTER, font="Monospace 14")
        self.lbl_answer.pack(side=LEFT, padx=10, pady=10)

        self.brush_size = 3
        self.brush_color = "black"
        self.canvas.bind("<B1-Motion>", self.draw)

        self.frame_bottom = Frame(self, relief=SUNKEN)
        self.frame_bottom.grid(row=2, column=0, columnspan=3, sticky=E + W + S + N)

        self.lbl_epoch = Label(self.frame_bottom)
        self.lbl_epoch.pack(side=LEFT, padx=5, pady=5)

        self.pb_train = Progressbar(
            self.frame_bottom, orient=HORIZONTAL,
            length=self.progress_bar_length,
            mode='determinate'
        )
        self.pb_train.config(mode="determinate", maximum=100, value=0)
        self.pb_train.pack(side=LEFT, padx=5, pady=5)

    def set_train_btn_listener(self, fun):
        self.btn_train.bind('<Button-1>', fun)

    def set_predict_listener(self, fun):
        self.btn_predict.bind('<Button-1>', fun)

    def show_train_pg(self, flag: bool):
        if flag:
            self.frame_bottom.grid(row=2, column=0, columnspan=3, sticky=E + W + S + N)
        else:
            self.frame_bottom.grid_forget()
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
        self.lbl_epoch.config(text=self.__TEXT_LABEL_EPOCH__ % value)
        self.lbl_epoch.update()

    def reload_pb(self):
        self.set_epoch(1)
        self.is_first_epoch = True
        self.set_pd(0)
        self.progress = 0

    def set_pd(self, value):
        self.progress += value
        self.pb_train.step(value)
        self.pb_train.update()

    def draw(self, event):
        self.canvas.create_oval(event.x - self.brush_size,
                                event.y - self.brush_size,
                                event.x + self.brush_size,
                                event.y + self.brush_size,
                                fill=self.brush_color, outline=self.brush_color)

    def save_image(self, path):
        self.canvas.update()
        self.canvas.postscript(file=path, colormode="mono")

    def enable_buttons(self, enable: bool):
        self.btn_train.config(state=(NORMAL if enable else DISABLED))
        self.btn_predict.config(state=(NORMAL if enable else DISABLED))
        self.btn_clear.config(state=(NORMAL if enable else DISABLED))

    def set_answer(self, value: int):
        self.lbl_answer.config(text=self.__TEXT_LABEL_ANSWER % value)
        self.lbl_answer.update()

    is_first_epoch = True
    epoch_count = 1
    progress = 0
    progress_bar_length = 100
    canvas_size = 200
    __TITLE__ = "Digit recognition"
    __TEXT_BTN_TRAIN__ = "Train"
    __TEXT_BTN_CLEAR__ = "Clear"
    __TEXT_BTN_PREDICT__ = "Predict"
    __TEXT_LABEL_EPOCH__ = "Epoch №%d"
    __TEXT_LABEL_ANSWER = "Your digit is %d."
