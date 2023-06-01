#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 18:34
# @Author  : CaoQixuan
# @File    : ImageRegionNet.py
# @Description : 获得每个图片的原始特征向量
# 该类是通过输入的标准型图片，进行多个分region后通过resnet网络得到向量后平均，生成图片模态向量

from torch import nn

from codes.Function import Load_ResNet50


class ImageRegionNet(nn.Module):

    def __init__(self, block_num=196, kernel_size=64, stride=32, output_size=2048, in_channel=3, device="gpu"):
        super().__init__()
        self.output_size = output_size  # 输出特征向量长度
        self.net = Load_ResNet50()  # 网络
        self.block_num = block_num  # 生成块数
        self.kernel_size = kernel_size  # 块的大小
        self.stride = stride  # 步长
        self.in_channel = in_channel  # 输入通道数

    def forward(self, X):
        batch_size, in_channel = X.shape[0], X.shape[1]
        # print("input_size: ", inputs.shape)
        output = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride)(X)
        # print("unfold_size: ", output.shape)
        output = output.transpose(1, 2).reshape(-1, in_channel, self.kernel_size,
                                                self.kernel_size)  # 一个图片划分为多个Region (batch_size * block_num, channel, kernel_size, kernel_size)

        output = self.net.forward(output).reshape(batch_size,
                                                  self.block_num,
                                                  self.output_size)  # 输入resNet网络后得到 (batch_size, block_num, h, w)

        return output  # 返回向量
