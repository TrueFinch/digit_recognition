import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from typing import Callable
from utils import check_file_exists, get_cwd
from PIL import Image
import utils


class TrainHistory(keras.callbacks.Callback):
    on_epoch_begin_callback: Callable[[], int] = None
    on_batch_begin_callback: Callable[[], int] = None
    on_train_end_callback: Callable[[], int] = None

    def on_batch_end(self, batch, logs=None):
        if self.on_batch_begin_callback is None:
            return
        self.on_batch_begin_callback()

    def on_train_end(self, logs=None):
        if self.on_train_end_callback is None:
            return
        self.on_train_end_callback()

    def on_epoch_begin(self, epoch, logs=None):
        if self.on_epoch_begin_callback is None:
            return
        self.on_epoch_begin_callback()


# class PredictCallback():

class MyModel:
    def __init__(self):
        # check if model exist
        # if exist -> load else -> train
        self.batch_size = 200
        self.num_classes = 10
        self.epochs = 20

        img_rows = img_cols = 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data(get_cwd() + "/.keras/datasets/mnist.npz")

        self.x_train = x_train.reshape(60000, 28, 28, 1)
        self.x_test = x_test.reshape(10000, 28, 28, 1)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train_samples')
        print(x_test.shape[0], 'test_samples')

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.model = None
        if check_file_exists(get_cwd() + "/.keras/models/model.h5") and check_file_exists(
                get_cwd() + "/.keras/models/model.yaml"):
            yaml_file = open(get_cwd() + "/.keras/models/model.yaml", "r")
            self.model: Sequential = keras.models.model_from_yaml(yaml_file.read())
            yaml_file.close()
            self.model.load_weights(get_cwd() + "/.keras/models/model.h5")

    def train(self, pb_history: TrainHistory):
        self.model = Sequential()
        # ===
        # self.model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        # self.model.add(MaxPooling2D())
        # self.model.add(Conv2D(15, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D())
        # self.model.add(Dropout(0.2))
        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dense(50, activation='relu'))
        # self.model.add(Dense(self.num_classes, activation='softmax'))
        # ===
        # self.model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        # self.model.add(MaxPooling2D())
        # self.model.add(Dropout(0.2))
        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dense(self.num_classes, activation='softmax'))
        # ====
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy']
        )
        self.model.add(Dense(self.num_classes, activation='softmax'))
        try:
            self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=2,
                validation_data=(self.x_test, self.y_test),
                callbacks=[pb_history]
            )
        except KeyboardInterrupt:
            print('\nTraining interrupted manually\n', flush=True)

        score = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        self.model.summary()

        with open(get_cwd() + "/.keras/models/model.yaml", "w+") as yaml_file:
            yaml_file.write(self.model.to_yaml())
        self.model.save_weights(get_cwd() + "/.keras/models/model.h5")
        print("Model was saved to disk.")

    def predict(self, path: str):
        # img = image.img_to_array(image.load_img(path, target_size=(28, 28)))
        # img = np.expand_dims(img, axis=0)
        # img = img.reshape(28, 28)
        img: Image = Image.open(path).resize((28, 28))
        arr = utils.rgb2gray(np.array(img))
        print(arr.shape)
        arr = arr.reshape(1, 28, 28, 1)
        result = self.model.predict(arr)
        c = 0
        for i in range(len(result[0])):
            if result[0][i] > result[0][c]:
                c = i
        print(str(c))
        pass
