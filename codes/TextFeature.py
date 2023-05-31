#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 19:33
# @Author  : CaoQixuan
# @File    : TextFeature.py
# @Description :文本特征提取
import numpy
import torch
from torch import nn
from transformers import BertModel

from codes.Attention import AdditiveAttention
from codes.Function import modelWightsDir, try_gpu


class TextFeature_LSTM(nn.Module):
    def __init__(self, nHidden, seqLen, guideLen, textEmbeddingPath, numLayers=1, dropout=0, device="cpu"):
        """
        :param nHidden: 隐藏层
        :param seqLen:  步长
        :param guideLen: 引导向量的维度 - 物品类别嵌入后的维度
        :param textEmbeddingPath: 文本glove后的向量
        :param numLayers: 网络层数 - 构建深层网络结构
        :param dropout:
        """
        super(TextFeature_LSTM, self).__init__()
        self.nHidden = nHidden
        self.seqLen = seqLen
        self.guideLen = guideLen
        self.numLayers = numLayers
        self.dropout = dropout
        self.device = device
        self.embeddingArray = torch.Tensor(numpy.load(textEmbeddingPath))
        if device == "gpu":
            self.embeddingArray = self.embeddingArray.to(try_gpu())
        self.embSize = self.embeddingArray.shape[1]  # 向量后的大小
        self.vocabSize = self.embeddingArray.shape[0]  # 类表大小
        self.embedding = nn.Embedding(self.vocabSize, self.embSize).from_pretrained(self.embeddingArray)
        self.layerNorm = nn.LayerNorm(self.embSize)
        self.fwLinearH = torch.nn.Linear(guideLen, self.nHidden)
        self.fwLinearC = torch.nn.Linear(guideLen, self.nHidden)
        self.bwLinearH = torch.nn.Linear(guideLen, self.nHidden)
        self.bwLinearC = torch.nn.Linear(guideLen, self.nHidden)
        self.relu = torch.nn.ReLU()
        self.biLSTM = nn.LSTM(input_size=self.embSize,
                              hidden_size=self.nHidden,
                              batch_first=True,
                              num_layers=self.numLayers,
                              dropout=self.dropout,
                              # 没有加dropout 因为不知道如何在测试时取消 20230426 - 预测时model.eval() / model.train() 来控制 20230427
                              bidirectional=True)

    @staticmethod
    def weight_init(m):
        # 默认方法
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, X, guideVector):
        """
        :param X:
        :param guideVector:  引导向量的 - 物品类别嵌入后
        :return: (batch_size, 2 * nHidden)
        """
        X = self.embedding(X)
        X = self.layerNorm(X)
        fw_h0 = self.relu(self.fwLinearH(guideVector))
        fw_c0 = self.relu(self.fwLinearC(guideVector))
        bw_h0 = self.relu(self.bwLinearH(guideVector))
        # bw_h0 = self.relu(self.fwLinearH(guideVector)) # 这里是否使用同一感知机层初始化正向和反向的H，C，需要进一步实验 20230427
        bw_c0 = self.relu(self.bwLinearC(guideVector))
        # bw_c0 = self.relu(self.fwLinearC(guideVector))
        init_h0 = torch.stack((fw_h0,) * self.numLayers + (bw_h0,) * self.numLayers,
                              dim=0)  # 深层LSTM是初始化为(D * layer , nHidden) -> (D, layers, nHidden) 观察API得出 存疑20230427
        init_c0 = torch.stack((fw_c0,) * self.numLayers + (bw_c0,) * self.numLayers,
                              dim=0)  # 加入stack 后网络的感知层是否会更新？ 20230427
        output, (_, _) = self.biLSTM(X, (init_h0, init_c0))  # output = batch_size * seqLen * (2 * hidden)
        return output, torch.mean(output, dim=1)  # batch_size * seqLen * (2 * hidden), batch_size * (2 * hidden)


class TextFeature_Bert(nn.Module):
    def __init__(self, nHidden, sqLen, dropout, device="cpu"):
        """
        :param nHidden:
        :param sqLen:
        :param dropout:
        :param device:
        """
        super(TextFeature_Bert, self).__init__()
        self.device = device
        self.nHidden = nHidden
        self.sqLen = sqLen
        self.bert = BertModel.from_pretrained(modelWightsDir + "bert-base-cased")
        self.layerNorm = nn.LayerNorm(768)  # 模型一般是768 如果是别的自己改一下
        self.linear = nn.Linear(768, self.nHidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()  # 保持在[-1, 1]之间，与LSTM激活函数保持一致，便于融合

    @staticmethod
    def weight_init(m):
        # 默认方法
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, text):
        input_ids, token_type_ids, attention_mask = text
        with torch.no_grad():
            output = self.bert(input_ids=input_ids.squeeze(1), token_type_ids=token_type_ids.squeeze(1),
                               attention_mask=attention_mask.squeeze(1))[0].detach()  # 截断梯度更新
        output = self.layerNorm(output)
        output = self.tanh(self.linear(output))
        output = self.dropout(output)
        return output, torch.mean(output, dim=1)  # batch_size * seqLen * (2 * hidden), batch_size * (2 * hidden)


class TextFeature(nn.Module):
    def __init__(self, nHidden, seqLen, guideLen, textEmbeddingPath, numLayers=1, dropout=0, device="cpu"):
        super(TextFeature, self).__init__()
        self.nHidden = nHidden
        self.lstm = TextFeature_LSTM(nHidden, seqLen, textEmbeddingPath=textEmbeddingPath,
                                     numLayers=numLayers,
                                     guideLen=guideLen, dropout=dropout, device=device)
        self.bert = TextFeature_Bert(nHidden=nHidden, sqLen=seqLen, dropout=dropout)
        self.attentionLSTM = AdditiveAttention(query_size=nHidden * 2, key_size=nHidden * 2, dropout=dropout,
                                               num_hiddens=nHidden)
        self.elu = nn.ELU()
        self.lstm.apply(self.lstm.weight_init)
        self.bert.apply(self.bert.weight_init)
        self.attentionLSTM.apply(self.attentionLSTM.weight_init)

    def forward(self, reText, text, guideVector):
        lstm_o, lstm_vec = self.lstm.forward(reText, guideVector)
        bert_o, bert_vec = self.bert(text)
        lstm_o = self.attentionLSTM.forward(lstm_o, lstm_o, lstm_o)  # 自注意力机制
        o = (lstm_o + bert_o) / 2  # 均值融合
        output = self.elu(o)
        output_vec = (lstm_vec + bert_vec) / 2
        return output, output_vec
