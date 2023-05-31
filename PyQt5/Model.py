#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 20:28
# @Author  : CaoQixuan
# @File    : Model.py
# @Description :
import os
import pickle
import random

import numpy
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer

from codes.Function import textEmbeddingPath, classEmbeddingPath, modelWightsDir, Load_ResNet50, try_gpu, textPrefix
from codes.ImageRegionNet import ImageRegionNet
from codes.NNManager import Net


def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Model(nn.Module):
    def __init__(self, modelWeightPath, device="cpu", textDir=None):
        super(Model, self).__init__()
        self.nHidden = 256  # 隐藏层 - Bi-LSTM
        self.seqLen = 80  # 步长 - Bi-LSTM
        self.numLayers = 2  # 隐藏层层数
        self.batchSize = 64  # 批量
        self.maxClipping = 5  # 梯度裁剪
        self.normType = 2  # 梯度的范式
        self.dropout = 0  # DropOut层的概率 留取80%
        self.imageToTensor = standard_transforms.ToTensor()
        self.imageResize = transforms.Resize([480, 480])

        self.extractImageFeature = ImageRegionNet(net=Load_ResNet50())
        self.net = self.net = Net(self.nHidden, self.seqLen, dropout=self.dropout,
                                  classEmbeddingPath=classEmbeddingPath,
                                  textEmbeddingPath=textEmbeddingPath, device=device, numLayers=self.numLayers)
        if device == "gpu":
            self.net.to(try_gpu())
            self.extractImageFeature.to(try_gpu())
        self.net.load_state_dict(torch.load(modelWeightPath))
        set_seed()
        self.net.eval()
        self.extractImageFeature.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(modelWightsDir + "bert-base-cased")
        with open(textDir + "textVocab.py3", 'rb') as f:
            self.word2id = pickle.load(f)  # 词表

    def processText(self, sqLen, source):
        """
        :param sqLen:文本长度
        :param source:字符串
        :return:对应词表的对应SqLen长度
        """
        strs = source.split(" ")
        if len(strs) > sqLen:
            strs = strs[:sqLen]
        strs = numpy.array(strs)
        func = numpy.vectorize(lambda x: self.word2id[x] if x in self.word2id else self.word2id['<unk>'])
        return numpy.pad(func(strs), (0, sqLen - len(strs)))

    def forward(self, text, imageFilePath):
        # torch.Size([64, 80]) torch.Size([64, 196, 2048]) torch.Size([64, 5])
        # torch.Size([64, 1, 80]) torch.Size([64, 1, 80]) torch.Size([64, 1, 80])
        with torch.no_grad():
            reText = torch.tensor(self.processText(self.seqLen, text), dtype=torch.int32).unsqueeze(0)
            reWord = torch.tensor([0, 0, 0, 0, 0]).unsqueeze(0)  # 类别默认为零
            image = self.imageToTensor(Image.open(imageFilePath).convert("RGB")).unsqueeze(0)
            image = self.extractImageFeature(
                self.imageResize(image))  # 20230531 更改了ImageRegionNet.py eval no grad() ResNet.py bn->relu
            # print(reText.shape, image.shape, reWord.shape)
            encodedInput = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.seqLen,
                                          truncation=True)  # 模型编码
            input_ids, token_type_ids, attention_mask = encodedInput["input_ids"].unsqueeze(0), encodedInput[
                "token_type_ids"].unsqueeze(0), encodedInput["attention_mask"].unsqueeze(0)
            # print(input_ids.shape, token_type_ids.shape, attention_mask.shape)
            return self.net((
                reText, image, reWord,
                (input_ids, token_type_ids, attention_mask)
            )).to("cpu")



"""
日志
20230529 修改了网络 ExtractFeature forward return squeeze-> squeeze(1)
                  TextFeature_Bert forward squeeze-> squeeze(1)
"""
