#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 10:30
# @Author  : CaoQixuan
# @File    : ThreadModel.py
# @Description : 并发操作加载模型和界面
from threading import Thread


class ThreadModel(Thread):
    def __init__(self, func, args=()):
        '''
        :param func: 被测试的函数
        :param args: 被测试的函数的返回值
        '''
        super(ThreadModel, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self) -> None:
        self.result = self.func(*self.args)

    def getResult(self):
        try:
            return self.result
        except BaseException as e:
            return e.args[0]
