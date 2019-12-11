import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from typing import Callable
from utils import check_file_exists, get_cwd


class ProgressHistory(keras.callbacks.Callback):
    on_train_begin_callback: Callable[[], int] = None
    on_batch_begin_callback: Callable[[], int] = None

    def on_train_begin(self, logs=None):
        if self.on_train_begin_callback is None:
            return
        self.on_train_begin_callback()

    def on_batch_end(self, batch, logs=None):
        if self.on_batch_begin_callback is None:
            return
        self.on_batch_begin_callback()

class MyModel:
    def __init__(self):
        # check if model exist
        # if exist -> load else -> train
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 12

        img_rows = img_cols = 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data(get_cwd() + "/.keras/datasets/data/mnist.npz")

        self.x_train = x_train.reshape(60000, 28, 28, 1)
        self.x_test = x_test.reshape(10000, 28, 28, 1)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train_samples')
        print(x_test.shape[0], 'test_samples')

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.model = None

    def train(self, ):
        with self:
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten)
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy']
            )
            model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=(self.x_test, self.y_test)
            )
