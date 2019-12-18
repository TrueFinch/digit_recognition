import os.path
from PIL import Image
import numpy as np
import cv2

def get_cwd():
    return os.getcwd()


def check_file_exists(path) -> bool:
    if path == "" and path is None:
        return False
    elif os.path.exists(path):
        return True
    else:
        return False


def prepare_image(path: str):
    # im = cv2.imread(path, 0)
    # im2 = cv2.resize(im, (28, 28))
    # im = im2.reshape(28, 28, -1)
    # im = im.reshape(1, 1, 28, 28)
    # im = cv2.bitwise_not(im)
    # return im
    img = Image.open(path)
    # width, heigth = img.size

    # if width == 28 and heigth == 28:
    #     return
    # img = img.resize((28, 28))
    img.save(get_cwd() + "/.keras/images/prepared_image.png", "PNG")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
