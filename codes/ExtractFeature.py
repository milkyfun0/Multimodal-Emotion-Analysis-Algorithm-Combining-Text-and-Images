#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 16:13
# @Author  : CaoQixuan
# @File    : ExtractFeature.py
# @Description :图像向量部分
import numpy
import torch
from torch import nn

from codes.Attention import AdditiveAttention
from codes.Function import try_gpu


class ExtractFeature(nn.Module):
    def __init__(self, embeddingPath, device="cpu"):
        super().__init__()
        self.embeddingArray = torch.Tensor(numpy.load(embeddingPath))  # 1001 * 300 1001为1000个类和1个unk
        if device == "gpu":
            self.embeddingArray = self.embeddingArray.to(try_gpu())
        self.embSize = self.embeddingArray.shape[1]  # 向量后的大小
        self.vocabSize = self.embeddingArray.shape[0]  # 类表大小
        self.embedding = nn.Embedding(self.vocabSize, self.embSize).from_pretrained(self.embeddingArray)
        self.linear1 = nn.Linear(self.embSize, self.embSize // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.embSize // 2, 1)
        self.softmax = nn.Softmax()
        self.attention = AdditiveAttention(key_size=self.embSize, query_size=self.embSize,
                                           num_hiddens=self.embSize // 2)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, X):
        batch_size, classes = X.shape[0], X.shape[1]
        output1 = self.embedding(X)  # batch * 5 * 200
        # # 这里也可以用之前写的注意力机制 两个版本
        return output1, torch.mean(self.attention.forward(queries=output1, keys=output1, values=output1),
                                   dim=1).squeeze(1)
        # output2 = self.relu(self.linear1(output1))  # batch * 5 * 100
        # output3 = self.softmax(self.linear2(output2)).reshape(batch_size, 1, classes)  # batch * 1 * 5
        # return output1, torch.squeeze(output3 @ output1)  # batch * 5 * 100, batch * 200
