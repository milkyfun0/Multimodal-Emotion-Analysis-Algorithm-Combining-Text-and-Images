#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 11:11
# @Author  : CaoQixuan
# @File    : NNManager.py
# @Description :总体网络搭建

import torch
from torch import nn

from codes.Attention import MultiModalAttention, DotProductAttention
from codes.ExtractFeature import ExtractFeature
from codes.Function import try_gpu
from codes.ImageFeature import ImageFeature
from codes.TextFeature import TextFeature


class Net(nn.Module):

    def __init__(self, nHidden, seqLen, dropout=0, numLayers=1, classEmbeddingPath="..//ExtractWords/vector",
                 textEmbeddingPath="../words/vector", device="cpu"):
        super().__init__()
        self.FinalMLPSize = 512
        self.device = device
        self.extractFeature = ExtractFeature(embeddingPath=classEmbeddingPath, device=device)  # 图像中物品类别
        self.imageFeature = ImageFeature()  # 图像特征
        self.imageFeature.apply(ImageFeature.weight_init)
        self.textFeature = TextFeature(nHidden, seqLen, textEmbeddingPath=textEmbeddingPath,
                                       numLayers=numLayers,
                                       guideLen=self.extractFeature.embSize, dropout=dropout, device=device)

        # 注意力机制以 x, y, z 指导向量计算与 key的评分，最后将其平均 这里用的是加性注意力机制，由seqToSeq翻译的注意力所启发
        self.extractFeatureATT = MultiModalAttention(
            querySizes=(
                self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
            keySize=self.extractFeature.embSize, dropout=dropout)
        self.imageFeatureATT = MultiModalAttention(
            querySizes=(
                self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
            keySize=self.imageFeature.defaultFeatureSize, dropout=dropout)
        self.textFeatureATT = MultiModalAttention(
            querySizes=(
                self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
            keySize=self.textFeature.nHidden * 2, dropout=dropout)

        # 为了后面的缩放点积注意力，需要把多模态向量调整为同一维度，后加入注意力机制，减少模型复杂度
        self.extractLinear = nn.Linear(self.extractFeature.embSize, self.FinalMLPSize)
        self.extractRelu = nn.ReLU()
        self.imageLinear = nn.Linear(self.imageFeature.defaultFeatureSize, self.FinalMLPSize)
        self.imageRelu = nn.ReLU()
        self.textLinear = nn.Linear(self.textFeature.nHidden * 2, self.FinalMLPSize)
        self.textRelu = nn.ReLU()

        self.multiAttention = DotProductAttention(dropout=dropout)

        # 最后加入两层全连接层
        self.MLP, self.FC = nn.Linear(self.FinalMLPSize, self.FinalMLPSize // 2), nn.Linear(self.FinalMLPSize // 2, 1)
        self.mlpRelu, self.fcSigmoid = nn.ReLU(), nn.Sigmoid()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, X):
        reText, images, reWords, text = X
        input_ids, token_type_ids, attention_mask = text

        if self.device == "gpu":
            reText, images, reWords = reText.to(try_gpu()), images.to(try_gpu()), reWords.to(
                try_gpu())
            input_ids, token_type_ids, attention_mask = input_ids.to(try_gpu()), token_type_ids.to(
                try_gpu()), attention_mask.to(try_gpu())

        extractMatrix, extractGuidVec = self.extractFeature.forward(reWords)
        imageMatrix, imageGuidVec = self.imageFeature.forward(images)
        textHMatrix, textGuidVec = self.textFeature.forward(reText, (input_ids, token_type_ids, attention_mask),
                                                            extractGuidVec)
        extractGuidVec, imageGuidVec, textGuidVec = extractGuidVec.unsqueeze(1), imageGuidVec.unsqueeze(
            1), textGuidVec.unsqueeze(1)  # 升维
        extractVec = self.extractFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), extractMatrix)
        imageVec = self.imageFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), imageMatrix)
        textVec = self.textFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), textHMatrix)  # QVA

        extractVec, imageVec, textVec = extractVec.squeeze(1), imageVec.squeeze(1), textVec.squeeze(1)  # 降维

        # 是否加入relu继续激活 未实验 20230504
        extractVec = self.extractLinear.forward(extractVec)
        extractVec = self.extractRelu(extractVec)
        imageVec = self.imageLinear.forward(imageVec)
        imageVec = self.imageRelu(imageVec)
        textVec = self.textLinear.forward(textVec)
        textVec = self.textRelu(textVec)
        finalMatrix = torch.stack((extractVec, imageVec, textVec), dim=1)  # 转化为 batch * 3 * FinalMLPSize
        finalVec = torch.mean(self.multiAttention.forward(finalMatrix), dim=1)
        fcInput = self.mlpRelu(self.MLP(finalVec))
        return self.fcSigmoid(self.FC(fcInput))
