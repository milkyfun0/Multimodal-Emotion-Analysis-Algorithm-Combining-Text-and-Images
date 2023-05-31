#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 15:57
# @Author  : CaoQixuan
# @File    : TimeWindow.py
# @Description : 进度条展示
from PyQt5.QtCore import QBasicTimer, pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel

from PyQt.Function import getQProgressBar, getFont


class TimePhase(QWidget):
    closeSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.resize(500, 300)
        font = getFont()
        self.pv = 0
        self.pgb = getQProgressBar(window=self,
                                   address=(100, 60),
                                   size=(300, 60),
                                   font=font,
                                   value=self.pv,
                                   range=[0, 100],
                                   des="Loading Model ")

        self.timer1 = QBasicTimer()
        self.label = QLabel('加载模型耐心等待', self)
        self.label.setFont(font)
        self.label.move(230, 150)

    def myTimerState(self):  # 开始
        if self.timer1.isActive():
            self.timer1.stop()
        else:
            self.timer1.start(100, self)

    def timerEvent(self, e):  # 加载
        if self.pv >= 98:
            self.timer1.stop()
        else:
            self.pv += 1.5
            self.pgb.setValue(self.pv)

    def closeEvent(self, event):  # 结束触发事件
        self.closeSignal.emit(str(self.pv))
