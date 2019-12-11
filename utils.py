import os.path


def get_cwd():
    return os.getcwd()


def check_file_exists(path) -> bool:
    if path == "" and path is None:
        return False
    elif os.path.exists(path):
        return True
    else:
        return False
