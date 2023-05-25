# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 19:20
# @Author  : CaoQixuan
# @File    : ImageFeature.py
# @Description :对于图片提取出来的向量加入网络
import torch
from torch import nn


class ImageFeature(nn.Module):
    def __init__(self, defaultFeatureSize=1024, device="cpu"):
        super().__init__()
        self.defaultFeatureSize = defaultFeatureSize
        self.linear = torch.nn.Linear(2048, self.defaultFeatureSize)
        self.relu = torch.nn.ReLU()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, X):
        output = self.relu(self.linear(X))
        return output, torch.mean(output, dim=1)  # batch * 196 *1024,  batch  * 1024

# if __name__ == "__main__":
