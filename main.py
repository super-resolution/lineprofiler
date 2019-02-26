# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from controllers.random_GUI import MainWindow
import tkinter as tk
import win32api

if __name__ == "__main__":
    qtApp = QApplication(sys.argv)
    qtWindow = QMainWindow()
    mainWindow = MainWindow()
    mainWindow.setupUi(qtWindow)

    mainWindow.init_component(qtWindow)
    qtWindow.show()
sys.exit(qtApp.exec_())