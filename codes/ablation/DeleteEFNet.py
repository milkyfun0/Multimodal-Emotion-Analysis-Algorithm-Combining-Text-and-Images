#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 15:22
# @Author  : CaoQixuan
# @File    : DeleteEFNet.py
# @Description :本网络是删除早期融合的网络
import time

import torch

from codes.Function import try_gpu, classEmbeddingPath, textEmbeddingPath
from codes.Main import Main
from codes.NNManager import Net


class DeleteEFNet(Net):
    """删除早期融合时需要在数据集方面取消字符串的直接concat，这里只是在网络中进行修改"""

    def __init__(self, nHidden, seqLen, dropout=0, numLayers=1, classEmbeddingPath="..//ExtractWords/vector",
                 textEmbeddingPath="../words/vector", device="cpu"):
        super(DeleteEFNet, self).__init__(nHidden=nHidden, seqLen=seqLen, dropout=dropout, numLayers=numLayers,
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
        # 改动 -------------------------------------------------------------------->
        extractGuidVec = torch.zeros_like(extractGuidVec)  # 这里使输入为零使早期融合无效
        # <---------------------------------------------------------------------
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


class MainDeleteEFNet(Main):
    def __init__(self, device="cpu"):
        super(MainDeleteEFNet, self).__init__(device=device, isJoin=False)
        self.net = DeleteEFNet(
            nHidden=self.nHidden,
            seqLen=self.seqLen,
            dropout=self.dropout,
            numLayers=self.numLayers,
            classEmbeddingPath=classEmbeddingPath,
            textEmbeddingPath=textEmbeddingPath,
            device=device
        )
        self.lr = 1e-5
        if device == "gpu":
            self.net.to(try_gpu())

    def saveNet(self, saveName=time.strftime("DeleteEFNet-%Y-%m-%d", time.localtime()), describe="unKnown"):
        super(MainDeleteEFNet, self).saveNet(saveName)

    def loadNet(self, loadName=time.strftime("DeleteEFNet-%Y-%m-%d", time.localtime()), isEval=False):
        super(MainDeleteEFNet, self).loadNet(loadName)


if __name__ == "__main__":
    print("<---------- 开始运行 祈祷不报错 ---------->")
    main = MainDeleteEFNet(device="gpu")
    main.train()
    print("<---------- 运行结束 万幸 --------------->")
