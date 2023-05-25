# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 9:58
# @Author  : CaoQixuan
# @File    : Attention.py
# @Description :各种注意力机制
import math

import torch
import torch.nn.functional
from torch import nn


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """
     通过在最后一个轴上掩蔽元素来执行softmax操作
    :param X: X:3D张量
    :param valid_lens: valid_lens:1D或2D张量
    :return:
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout=0):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def weight_init(m):
        # 默认方法
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, queries, keys, values, valid_lens=None):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batchSize，查询的个数，1，num_hidden)
        # key的形状：(batchSize，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        # print(queries.unsqueeze(2).shape, keys.unsqueeze(1).shape)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batchSize，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batchSize，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.attention_weights = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, valid_lens=None):
        """
        :param valid_lens: 忽略某些键值对时用
        :param X: (batchSize，查询的个数，d)
        :return: batchSize * 值的维度
        """
        #  queries: (batchSize，查询的个数，d)
        #  keys: (batchSize，“键－值”对的个数，d)
        #  values: (batchSize，“键－值”对的个数，值的维度)
        #  valid_lens: (batchSize，查询的个数)
        queries, keys, values = X, X, X
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)  # batchSize，查询个数的个数，d


class MultiModalAttention(nn.Module):
    """多模态加性注意力机制融合"""

    def __init__(self, querySizes, keySize, dropout=0):
        """
        QKV: query = query, key=key, value=key
        :param querySizes: 利用多个向量进行融合-源于
        :param keySize:
        :param dropout:
        """
        super().__init__()
        self.attentions = []
        count = 0
        for querySize in querySizes:
            exec(
                "self.addATT_{} = AdditiveAttention(query_size=querySize, key_size=keySize, num_hiddens=keySize // 2, "
                "dropout=dropout)".format(
                    count))
            exec("self.attentions.append(self.addATT_{})".format(count))
            count += 1
        [attention.apply(MultiModalAttention.weight_init) for attention in self.attentions]

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, queries, key):
        """
        :param queries: (query1, query2...)
        :param key: batchSize, 键值对, values
        :return: batchSize * 1 * 值的维度
        """
        vector = torch.zeros(key.shape[0], 1, key.shape[-1], device=key.device)  # 这里加维为了后面stack
        for attention, query in zip(self.attentions, queries):
            vector += attention.forward(queries=query, keys=key, values=key)
        return vector / len(self.attentions)
