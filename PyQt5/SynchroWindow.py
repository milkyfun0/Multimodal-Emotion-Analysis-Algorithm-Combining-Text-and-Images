#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 21:10
# @Author  : CaoQixuan
# @File    : SynchroWindow.py
# @Description : 同步timeWindow和disPlayWindow

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from PyQt.Model import Model
from codes.Function import textPrefix


class SynchroWindow(QWidget):
    closeSignal = pyqtSignal(float)

    def __init__(self, model=None):
        super(SynchroWindow, self).__init__()
        self.model = None
        self.P = 0

    def predict(self, text, imageFilePath, modelWeightPath="./modelWeight/model.pth"):
        # 默认加载codes产生的模型
        if self.model is None:
            self.model = Model(modelWeightPath, textDir=textPrefix)
        self.P = self.model(text, imageFilePath)[0][0]
        self.close()

    def closeEvent(self, event):  # 关闭触发事件
        self.closeSignal.emit(self.P)
