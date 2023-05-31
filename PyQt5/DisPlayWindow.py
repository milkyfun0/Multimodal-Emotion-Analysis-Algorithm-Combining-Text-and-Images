#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 15:58
# @Author  : CaoQixuan
# @File    : DisPlayWindow.py
# @Description : 展示结果
import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QTextBrowser, QLabel

from PyQt.Function import getFont, getQProgressBar


class DisPlayWindow(QWidget):
    closeSignal = pyqtSignal(str)

    def __init__(self, text="Hello Word", imageFilePath=None, value=50, des=""):
        super().__init__()
        self.resize(800, 600)
        font = getFont()
        self.setWindowTitle('检测结果')

        self.imageQ = QLabel(self)
        self.imageQ.setGeometry(150, 30, 500, 300)
        if imageFilePath is not None:
            self.imageQ.setPixmap(QPixmap(imageFilePath).scaledToWidth(400))
        else:
            des = "NOT FIND IMAGE "
            value = 0

        self.textQ = QTextBrowser(self)
        self.textQ.setPlainText("  " + text)
        self.textQ.setFont(font)
        self.textQ.setGeometry(100, 300, 600, 120)

        self.P = getQProgressBar(
            window=self,
            address=(230, 450),
            size=(300, 60),
            font=font,
            value=value,
            range=[0, 100],
            des=des,
            color="(34,139,34)"
        )
        label = QLabel("存在反讽的概率：", self)
        label.setGeometry(80, 460, 140, 40)
        font.setPointSize(13)
        font.setFamily("黑体")
        label.setFont(font)

    def closeEvent(self, event):  # 关闭时触发事件
        self.closeSignal.emit("exit")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DisPlayWindow()
    window.show()
    app.exec_()
