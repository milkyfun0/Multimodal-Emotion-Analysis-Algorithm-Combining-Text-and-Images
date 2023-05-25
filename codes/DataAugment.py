#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 9:58
# @Author  : CaoQixuan
# @File    : DataAugment.py
# @Description :
import os
import pickle
import random
import shutil

import numpy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm


def augmentBatch(image, text, id, imageInsert, textInsert, idsInsert, sqLen=10, alpha=0.7):
    """
    :param image:base
    :param text:base
    :param id:base
    :param imageInsert:insert
    :param textInsert: insert
    :param idsInsert:insert
    :param sqLen:
    :param alpha:保留率 (0, 1)
    需要进一步优化，增加并行度
    :return:
    """
    batch_size = image.size()[0]
    index = numpy.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        image[i, :] = alpha * image[i, :] + (1 - alpha) * imageInsert[index[i], :]
        # text random insert
        base = text[i].split(" ")
        insert = textInsert[index[i]].split(" ")
        num = min(len(insert), int(len(base) * (1 - alpha)), int(sqLen * (1 - alpha)))  # 取最小额度
        randIndex = random.sample(range(len(insert)), num)
        words = numpy.array(insert)[randIndex]
        randIndex = random.sample(range(len(base)), num)
        for j in range(num):
            base.insert(randIndex[j], words[j])
        text[i] = " ".join(base)
        id[i] = id[i] + "-" + idsInsert[index[i]]
    return id, image, text


class DataAugment:
    class DataSet(Dataset):
        def __init__(self, imgDir, textPath):
            super().__init__()
            self.imgDir = imgDir
            self.id2text = []
            with open(textPath, 'r', encoding="utf-8") as f:
                for line in f:
                    self.id2text.append(eval(line))
            self.id2text = numpy.array(self.id2text, dtype="object")
            self.transform = transforms.Compose([
                transforms.ToTensor()  # 这里仅以最基本的为例
            ])

        def __getitem__(self, item):
            id, text, y = self.id2text[item][0], self.id2text[item][1], self.id2text[item][2]
            img = self.transform(Image.open(self.imgDir + id + "/" + id + ".jpg").convert('RGB'))  # 不同的格式不一样自己改
            return id, img, text, y

        def __len__(self):
            return len(self.id2text)

    def __init__(self, seqLen, batchSize, imgDir, textPath, image2ClassPath):
        self.image2ClassPath = image2ClassPath
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.dataLoader = DataLoader(dataset=self.DataSet(imgDir=imgDir, textPath=textPath)
                                     , batch_size=batchSize, shuffle=True)

    def start(self, rate, imgSaveDir, image2ClassSaveDir, alpha=0.7):
        """
        :param rate: 增强占数据集的比例
        :param image2ClassSaveDir:
        :param imgSaveDir:
        :param alpha: 保留原图像的强度
        :return:
        """
        countBatch = 0
        num = int(rate * len(self.dataLoader))
        if os.path.exists(imgSaveDir):
            shutil.rmtree(imgSaveDir)
        os.makedirs(imgSaveDir, exist_ok=True)
        textSave = []
        with open(self.image2ClassPath, "rb") as f:
            image2Class = pickle.load(f)
        for X, i in zip(self.dataLoader, tqdm(range(num))):
            ids, images, texts, y = X
            idsAug, images, texts = augmentBatch(image=images[0:self.batchSize // 2],
                                                 text=list(texts[:self.batchSize // 2]),
                                                 id=list(ids[:self.batchSize // 2]),
                                                 imageInsert=images[self.batchSize // 2:],
                                                 textInsert=list(texts[self.batchSize // 2:]),
                                                 idsInsert=list(ids[self.batchSize // 2:]),
                                                 alpha=alpha,
                                                 )
            for j in range(self.batchSize // 2):
                os.mkdir(imgSaveDir + idsAug[j] + "/")
                utils.save_image(images[j], imgSaveDir + idsAug[j] + "/" + idsAug[j] + ".jpg")
                textSave.append(str([idsAug[j], texts[j], int(y[j])]))
                image2Class[idsAug[j]] = image2Class[ids[j]]
            if countBatch >= num:
                break
            countBatch += 1
        with open(image2ClassSaveDir + "image2class.py3", "wb+") as f:
            pickle.dump(image2Class, f)
        print(" 已增强数据增强数目", self.batchSize * int(rate * len(self.dataLoader)))
        return textSave
