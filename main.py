from tkinter import *
from window import MainWindow


def main() -> int:
    root = Tk()
    root.geometry("250x150+300+300")
    app = MainWindow(root)
    app.__TITLE__ = ""
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
