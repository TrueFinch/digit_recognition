import os.path
from PIL import Image


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
    img = Image.open(path)
    img.save(get_cwd() + "/.keras/images/prepared_image.png", "PNG")
