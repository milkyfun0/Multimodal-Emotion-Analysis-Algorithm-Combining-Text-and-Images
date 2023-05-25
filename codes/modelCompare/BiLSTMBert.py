#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 15:39
# @Author  : CaoQixuan
# @File    : BiLSTMBert.py
# @Description :仅保留文本特征的网络
import time

import torch
from torch import nn

from codes.Function import try_gpu, classEmbeddingPath, textEmbeddingPath
from codes.Main import Main
from codes.NNManager import Net


class BiLSTMBert(Net):

    def __init__(self, nHidden, seqLen, dropout=0, numLayers=1, classEmbeddingPath="..//ExtractWords/vector",
                 textEmbeddingPath="../words/vector", device="cpu"):
        super(BiLSTMBert, self).__init__(nHidden=nHidden, seqLen=seqLen, dropout=dropout, numLayers=numLayers,
                                         classEmbeddingPath=classEmbeddingPath, textEmbeddingPath=textEmbeddingPath,
                                         device=device)

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
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
        # 改动 ---------------------------------------------------------------------------------------------->
        _, extractGuidVec = self.extractFeature.forward(reWords)
        extractGuidVec = torch.zeros_like(extractGuidVec)
        _, textGuidVec = self.textFeature.forward(reText, (input_ids, token_type_ids, attention_mask),
                                                  extractGuidVec)
        textVec = self.textLinear.forward(textGuidVec)
        textVec = self.textRelu(textVec)
        finalMatrix = torch.stack(textVec, dim=1)  # 转化为 batch * 1 * FinalMLPSize
        # <----------------------------------------------------------------------------------------------------
        finalVec = torch.mean(self.multiAttention.forward(finalMatrix), dim=1)
        fcInput = self.mlpRelu(self.MLP(finalVec))
        return self.fcSigmoid(self.FC(fcInput))


class MainBiLSTMBert(Main):
    def __init__(self, device="cpu"):
        super(MainBiLSTMBert, self).__init__(device=device, isJoin=True)
        self.net = BiLSTMBert(
            nHidden=self.nHidden,
            seqLen=self.seqLen,
            dropout=self.dropout,
            numLayers=self.numLayers,
            classEmbeddingPath=classEmbeddingPath,
            textEmbeddingPath=textEmbeddingPath,
            device=device
        )
        if device == "gpu":
            self.net.to(try_gpu())

    def saveNet(self, saveName=time.strftime("DeleteRFNet-%Y-%m-%d", time.localtime()), describe="unKnown"):
        super(MainBiLSTMBert, self).saveNet(saveName)

    def loadNet(self, loadName=time.strftime("DeleteRFNet-%Y-%m-%d", time.localtime()), isEval=False):
        super(MainBiLSTMBert, self).loadNet(loadName, isEval=isEval)


if __name__ == "__main__":
    main = MainBiLSTMBert(device="gpu")
    main.train()
