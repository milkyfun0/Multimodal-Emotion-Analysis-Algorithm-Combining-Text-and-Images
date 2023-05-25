#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 19:24
# @Author  : CaoQixuan
# @File    : DATASET.py
# @Description :
from enum import Enum


class DATASET(Enum):
    TRAIN = "train_text"
    TEST = "test_text"
    VALID = "valid_text"
