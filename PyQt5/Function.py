#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 19:40
# @Author  : CaoQixuan
# @File    : Function.py
# @Description : 常用静态函数
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QProgressBar


def getQProgressBar(window, address, size, font, value, range, des, color="(100,200,200)"):
    # 设置一个值表示进度条的当前进度
    # 申明一个时钟控件
    pgb = QProgressBar(window)
    pgb.move(*address)
    pgb.resize(*size)
    pgb.setStyleSheet(
        "QProgressBar { "
        "border: 2px solid grey; "
        "border-radius: 5px; color: rgb(20,20,20); "
        " background-color: #FFFFFF; "
        "text-align: center;"
        "}"
        "QProgressBar::chunk {"
        "background-color: rgb" + color + "; "
                                          "border-radius: 10px; "
                                          "margin: 0.1px;  "  ## margin 设置两步之间的间隔
                                          "width: 2px;}")  ## 其中 width 是设置进度条每一步的宽度
    pgb.setFont(font)
    pgb.setValue(value)
    pgb.setMinimum(range[0])
    pgb.setMaximum(range[1])
    pgb.setFormat(des + '%p%'.format(pgb.value() - pgb.minimum()))
    return pgb


def getFont(name="Times New Roman", bold=True, size=15, weight=100):
    font = QFont()
    font.setFamily(name)
    font.setBold(bold)
    font.setPointSize(size)
    font.setWeight(weight)
    return font
