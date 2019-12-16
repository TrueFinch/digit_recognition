from tkinter import *
from window import MainWindow
from digit_recognition import MyModel, TrainHistory
import utils


def main() -> int:
    root = Tk()
    root.geometry("300x170+300+300")
    root.resizable(False, False)
    app = MainWindow(root)
    model = MyModel()

    def train_btn_listener(event):
        app.train_btn.config(state=DISABLED)
        app.predict_btn.config(state=DISABLED)
        app.clear_btn.config(state=DISABLED)
        app.reload_pb()
        app.show_train_pg(True)
        model.train(pb_history)

    def train_btn_callback():
        app.train_btn.config(state=NORMAL)
        app.predict_btn.config(state=NORMAL)
        app.clear_btn.config(state=NORMAL)
        app.show_train_pg(False)

    def predict_btn_listener(event):
        app.save_image(utils.get_cwd() + "/.keras/images/orig_image.ps")
        utils.prepare_image(utils.get_cwd() + "/.keras/images/orig_image.ps")
        model.predict(utils.get_cwd() + "/.keras/images/prepared_image.png")

    def predict_btn_callback():
        pass

    pb_history = TrainHistory()
    pb_history.on_epoch_begin_callback = app.increase_epoch
    pb_history.on_batch_begin_callback = lambda: app.increase_pg(model.batch_size * 100 / model.x_train.shape[0])
    pb_history.on_train_end_callback = lambda: train_btn_callback()

    app.set_train_btn_listener(train_btn_listener)
    app.set_predict_callback(predict_btn_listener)
    app.show_train_pg(False)

    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
