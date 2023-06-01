#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 15:02
# @Author  : CaoQixuan
# @File    : MyDataSet.py
# @Description : 读取各种数据

import pickle
from operator import itemgetter

import numpy
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from codes.DATASET import DATASET
from codes.Function import modelWightsDir, flitIllWord, mapDataSet


class MyDataSet(Dataset):
    """
    用于模型训练的数据集
    """

    def __init__(self, seqLen, imageClassDir, imageDir, textDir, dataType=DATASET.TRAIN, isJoin=True):
        """
        :param seqLen:
        :param imageClassDir:图片对应类的字典序列化文件夹
        :param imageVectorDir:ResNet生成的文本向量的文件夹
        :param textDir:推特数据的文本数据文件夹
        :param isJoin:是否在concat 约为早期融合
        """
        self.isJoin = isJoin
        self.seqLen = seqLen
        self.imageDir = imageDir
        self.id2text = []
        self.imageToTensor = transforms.ToTensor()
        self.imageResize = transforms.Resize([480, 480])
        self.tokenizer = AutoTokenizer.from_pretrained(modelWightsDir + "bert-base-cased")
        with open(textDir + "textVocab.py3", 'rb') as f:
            self.word2id = pickle.load(f)  # 词表
        with open(imageClassDir + "classVocab.py3", 'rb') as f:
            self.attribute2id = pickle.load(f)  # 类表
        with open(imageClassDir + "image2class.py3", 'rb') as f:
            self.dictExtractWords = pickle.load(f)  # 类表
        with open(textDir + mapDataSet[dataType], 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                if flitIllWord(line):  # 过滤非法word 20230513
                    self.id2text.append(eval(line))
        self.id2text = numpy.array(self.id2text)

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

    def __getitem__(self, index):
        id = self.id2text[index][0]
        if self.isJoin:
            text = ' '.join(self.dictExtractWords[id]) + self.id2text[index][1]
        else:
            text = self.id2text[index][1]
        reText = torch.tensor(self.processText(self.seqLen, text), dtype=torch.int32)
        retY = torch.tensor(float(self.id2text[index][2]), dtype=torch.float32)
        reWords = torch.tensor(itemgetter(*self.dictExtractWords[id])(self.attribute2id))
        if "-" not in id:
            image = (Image.open(self.imageDir + "/imageDataSet/" + id + "/" + id + ".jpg").convert("RGB"))
        else:
            image = (Image.open(self.imageDir + "/augmentImages/" + id + "/" + id + ".jpg").convert("RGB"))
        image = self.imageResize(self.imageToTensor(image))
        encodedInput = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.seqLen,
                                      truncation=True)  # 模型编码
        input_ids, token_type_ids, attention_mask = encodedInput["input_ids"], encodedInput[
            "token_type_ids"], encodedInput["attention_mask"]
        return (reText, image, reWords, (input_ids, token_type_ids, attention_mask)), retY, id

    def __len__(self):
        return self.id2text.shape[0]
