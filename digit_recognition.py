import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from typing import Callable, Optional
from utils import check_file_exists, get_cwd
import cv2

seed = 25122019
np.random.seed(seed)

keras.backend.common.set_image_dim_ordering('th')
keras.backend.image_data_format()


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


class MyModel:
    def __init__(self):
        self.batch_size = 200
        self.num_classes = 10
        self.epochs = 50

        # check if model exist
        # if exist -> load else -> train
        (x_train, y_train), (x_test, y_test) = mnist.load_data(get_cwd() + "/.keras/datasets/mnist.npz")

        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = keras.utils.to_categorical(y_train)
        self.y_test = keras.utils.to_categorical(y_test)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train_samples')
        print(x_test.shape[0], 'test_samples')

        self.model = None
        if check_file_exists(get_cwd() + "/.keras/models/model.h5") and check_file_exists(
                get_cwd() + "/.keras/models/model.yaml"):
            yaml_file = open(get_cwd() + "/.keras/models/model.yaml", "r")
            self.model: Sequential = keras.models.model_from_yaml(yaml_file.read())
            yaml_file.close()
            self.model.load_weights(get_cwd() + "/.keras/models/model.h5")

    def train(self, pb_history: TrainHistory):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        # option to interrupt training with saving model
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

    def predict(self, path: str, predict_callback: Optional[Callable] = None):
        im = cv2.imread(path, 0)
        im2 = cv2.resize(im, (28, 28))
        im = im2.reshape(28, 28, -1)
        im = im.reshape(1, 28, 28, 1)
        im = cv2.bitwise_not(im)

        result = self.model.predict(im)
        a = np.argmax(result)
        print(a)
        print(result)
        if predict_callback is not None:
            predict_callback(a)
        pass
