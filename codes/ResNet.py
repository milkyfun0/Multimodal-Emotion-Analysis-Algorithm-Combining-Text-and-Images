#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 21:48
# @Author  : CaoQixuan
# @File    : ResNet.py
# @Description : 这个文件是用来获得预训练模型的 downsample变量不能改为其他名字，服了

import torch
from torch import nn


class Residual(nn.Module):
    """ 残差块 -50"""
    expansion = 4  # 残差块第3个卷积层的通道膨胀倍率

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None, use_1x1conv=False):
        """
        :param in_channel:残差块输入通道数
        :param out_channel:残差块输出通道数
        :param stride:卷积步长
        :param down_sample:在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
        这里的意思是 在整个卷积层的开始时，会发生 H/2 W/2
        :param use_1x1conv:
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
                               bias=False)  # H,W不变: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # H/2，W/2 C不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
                               stride=1, bias=False)  # H,W不变 C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(num_features=out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = down_sample

    def forward(self, X):
        X_res = X
        if self.downsample is not None:
            X_res = self.downsample(X_res)
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output += X_res  # 残差连接
        return self.relu(output)


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000):
        """
        :param block:堆叠的基本模块
        :param block_num:基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
        :param num_classes:num_classes: 全连接之后的分类特征维度
        """
        super(ResNet, self).__init__()
        self.in_channel = 64  # conv1的输出通道数
        # 网络开始 224 * 224-> 112 * 112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)  # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 网络开始 112 * 112-> 56 * 56
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.resnet_block(block=block, channel=64, block_num=block_num[0],
                                        stride=1)  # H W 不变 不需要下采样
        self.layer2 = self.resnet_block(block=block, channel=128, block_num=block_num[1],
                                        stride=2)  # H W 减半 50 101 150 需要下采样
        self.layer3 = self.resnet_block(block=block, channel=256, block_num=block_num[2],
                                        stride=2)  # H W 减半 50 101 150 需要下采样
        self.layer4 = self.resnet_block(block=block, channel=512, block_num=block_num[3],
                                        stride=2)  # H W 减半 50 101 150 需要下采样

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        for m in self.modules():  # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def resnet_block(self, block, channel, block_num, stride=1):
        """
        :param block: 堆叠的基本模块
        :param channel:基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
        :param block_num:当期stage堆叠block个数
        :param stride: 默认卷积步长
        :return: 生成的blocks
        """
        downsample = None  # 用于控制下采样的 即减半的
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),  # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                nn.BatchNorm2d(num_features=channel * block.expansion))

        blocks = []
        blocks.append(block(in_channel=self.in_channel, out_channel=channel, down_sample=downsample,
                            stride=stride))  # 定义convi_x中的第一个残差块，只有第一个需要设置down_sample和stride
        self.in_channel = channel * block.expansion  # 在下一次调用_make_layer函数的时候，self.in_channel已经x4
        for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
            blocks.append(block(in_channel=self.in_channel, out_channel=channel))
        return nn.Sequential(*blocks)

    def forward(self, X):
        output = self.max_pool(self.bn1(self.bn1(self.conv1(X))))

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        # output = self.fc(output)

        return output
