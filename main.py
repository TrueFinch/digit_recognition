from tkinter import *
from window import MainWindow


def main() -> int:
    root = Tk()
    root.geometry("250x150+300+300")
    root.resizable(False, False)
    app = MainWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()
