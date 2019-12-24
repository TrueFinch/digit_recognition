from tkinter import *
from window import MainWindow
from digit_recognition import MyModel, TrainHistory
import utils


def main() -> int:
    root = Tk()
    root.geometry("450x300+300+300")
    root.resizable(False, False)
    app = MainWindow(root)
    model = MyModel()

    def train_btn_listener(event):
        app.enable_buttons(False)
        app.reload_pb()
        app.show_train_pg(True)
        model.train(pb_history)

    def train_btn_callback():
        app.enable_buttons(True)
        app.show_train_pg(False)

    def predict_btn_callback(value: int):
        app.enable_buttons(True)
        app.set_answer(value)

    def predict_btn_listener(event):
        app.enable_buttons(False)
        app.save_image(utils.get_cwd() + "/.keras/images/orig_image.ps")
        utils.prepare_image(utils.get_cwd() + "/.keras/images/orig_image.ps")
        model.predict(utils.get_cwd() + "/.keras/images/prepared_image.png", predict_btn_callback)

    pb_history = TrainHistory()
    pb_history.on_epoch_begin_callback = app.increase_epoch
    pb_history.on_batch_begin_callback = lambda: app.increase_pg(model.batch_size * 100 / model.x_train.shape[0])
    pb_history.on_train_end_callback = lambda: train_btn_callback()

    app.set_train_btn_listener(train_btn_listener)
    app.set_predict_listener(predict_btn_listener)
    app.show_train_pg(False)

    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
