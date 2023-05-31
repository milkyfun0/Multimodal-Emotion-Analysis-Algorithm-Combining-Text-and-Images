#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/27 10:30
# @Author  : CaoQixuan
# @File    : MainWindow.py
# @Description : 主窗口


import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPlainTextEdit, QPushButton, QFileDialog

from PyQt.DisPlayWindow import DisPlayWindow
from PyQt.SynchroWindow import SynchroWindow
from PyQt.ThreadModel import ThreadModel
from PyQt.TimeWindow import TimePhase, getFont


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.text = None
        self.imgPixmap = None
        self.fileName = None
        self.model = None
        self.P = 0

        self.imgLabel = QLabel("请选择图片：", self)
        self.imageShow = QLabel(self)
        self.chooseButton = QPushButton("选择", self)
        self.textLabel = QLabel("请输入文本：", self)
        self.textEdit = QPlainTextEdit(self)
        self.submitButton = QPushButton("提交", self)
        self.disPlayWindow = None
        self.synchroWindow = SynchroWindow()
        self.thread = None
        self.timeWindow = TimePhase()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("My Window")
        self.resize(800, 600)
        # 图片标签和选择按钮
        self.imgLabel.setGeometry(50, 50, 100, 30)
        self.imgLabel.setFont(getFont(size=10))

        self.imageShow.setGeometry(200, 30, 500, 250)

        self.chooseButton.setFont(getFont(size=10))
        self.chooseButton.setGeometry(680, 150, 80, 30)
        self.chooseButton.clicked.connect(self.chooseImage)

        # 文本输入框
        self.textLabel.setGeometry(50, 300, 100, 30)
        self.textLabel.setFont(getFont(size=10))

        self.textEdit.setGeometry(150, 300, 600, 120)
        self.textEdit.setFont(getFont())

        # 提交按钮
        self.submitButton.setGeometry(350, 530, 100, 50)
        self.submitButton.clicked.connect(self.submit)
        self.submitButton.setFont(getFont())

        self.synchroWindow.closeSignal.connect(self.closeTimeWindow)
        self.timeWindow.closeSignal.connect(self.showDisplayWindow)

    def chooseImage(self):
        # 选择图片文件
        self.fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
        if self.fileName:
            self.imageShow.setPixmap(QPixmap(self.fileName).scaledToWidth(400))

    def submit(self):
        # 获取输入的文本和图片
        self.text = self.textEdit.toPlainText()
        if not self.fileName or not self.text:
            self.restartMainWindow("")
            return
        self.timeWindow.show()
        self.timeWindow.myTimerState()
        self.thread = ThreadModel(self.synchroWindow.predict, args=(self.text, self.fileName))
        self.thread.start()
        self.close()

    def closeTimeWindow(self, P):
        self.timeWindow.pgb.setValue(100)
        # 模型加载完毕后
        self.P = P
        self.timeWindow.close()

    def showDisplayWindow(self, info):
        self.disPlayWindow = DisPlayWindow(text=self.text, imageFilePath=self.fileName, value=int(self.P * 100))
        self.disPlayWindow.closeSignal.connect(self.restartMainWindow)
        self.disPlayWindow.show()

    def restartMainWindow(self, info):
        self.textEdit.clear()
        self.imageShow.clear()
        self.show()

    def keyPressEvent(self, event):
        # 如果按下回车键，自动触发提交按钮
        if event.key() == Qt.Key_Return:
            self.submit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
    print("Over")
