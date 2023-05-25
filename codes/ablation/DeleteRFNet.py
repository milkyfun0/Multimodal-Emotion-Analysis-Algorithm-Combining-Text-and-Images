#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 15:29
# @Author  : CaoQixuan
# @File    : DeleteRFNet.py
# @Description :删除了表示融合的网络
import time

import torch

from codes.Function import try_gpu, classEmbeddingPath, textEmbeddingPath
from codes.Main import Main
from codes.NNManager import Net


class DeleteRFNet(Net):

    def __init__(self, nHidden, seqLen, dropout=0, numLayers=1, classEmbeddingPath="..//ExtractWords/vector",
                 textEmbeddingPath="../words/vector", device="cpu"):
        super(DeleteRFNet, self).__init__(nHidden=nHidden, seqLen=seqLen, dropout=dropout, numLayers=numLayers,
                                          classEmbeddingPath=classEmbeddingPath, textEmbeddingPath=textEmbeddingPath,
                                          device=device)

    def forward(self, X):
        reText, images, reWords, text = X
        input_ids, token_type_ids, attention_mask = text

        if self.device == "gpu":
            reText, images, reWords = reText.to(try_gpu()), images.to(try_gpu()), reWords.to(
                try_gpu())
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()

        extractMatrix, extractGuidVec = self.extractFeature.forward(reWords)
        imageMatrix, imageGuidVec = self.imageFeature.forward(images)
        textHMatrix, textGuidVec = self.textFeature.forward(reText, (input_ids, token_type_ids, attention_mask),
                                                            extractGuidVec)
        # 改动 --------------------------------------------------------------------------------------->
        extractVec, imageVec, textVec = extractGuidVec, imageGuidVec, textGuidVec  # 降维
        # <----------------------------------------------------------------------------------------

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


class MainDeleteRFNet(Main):
    def __init__(self, device="cpu"):
        super(MainDeleteRFNet, self).__init__(device=device, isJoin=True)
        self.net = DeleteRFNet(
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
        super(MainDeleteRFNet, self).saveNet(saveName)

    def loadNet(self, loadName=time.strftime("DeleteRFNet-%Y-%m-%d", time.localtime()), isEval=False):
        super(MainDeleteRFNet, self).loadNet(loadName)


if __name__ == "__main__":
    main = MainDeleteRFNet(device="gpu")
    main.train()
