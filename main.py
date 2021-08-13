# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:28:30 2021

@author: elif
"""


import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from coding import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    