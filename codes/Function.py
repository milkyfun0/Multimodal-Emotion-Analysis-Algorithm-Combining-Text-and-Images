#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/5 10:50
# @Author  : CaoQixuan
# @File    : Function.py
# @Description :常用的函数 及其文件路径

import os
import pickle
import shutil
import time

import numpy
import torch
import torch.utils.data as data
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

from codes.DATASET import DATASET
from codes.DataAugment import DataAugment
from codes.ImageRegionNet import ImageRegionNet
from codes.ResNet import ResNet, Residual

illWords = ["sarcasm", "sarcastic", "reposting", "url", "joke", "humour", "humor", "jokes", "irony", "exgag"]
scoreNames = ["acc", "pre", "rec", "f1", "auc", "loss"]
row, col = 2, 3  # 对应scoreNames
mapDataSet = {
    DATASET.TRAIN: "train_text",
    DATASET.TEST: "test_text",
    DATASET.VALID: "valid_text"
}

# inputs path
# rootPath = os.path.abspath('..')
rootPath = "D:/Code/PyCharm/Multimodal/"
readImagesDir = rootPath + "/image/imageDataSet/"  # 原始图片

imagePrefix = rootPath + "/image/"  # 图片的存储目录
classPrefix = rootPath + "/class/"  # 类名对应的编号和GLove矩阵
textPrefix = rootPath + "/text/"  # 图片对应的文本 ["id", "text", "is_sarcasm"]，vocab，GLove矩阵

modelWightsDir = rootPath + "/modelWeights/"  # 模型权重
classEmbeddingPath = rootPath + "/class/classGloveVector.npy"  # 训练完成的类别嵌入矩阵
textEmbeddingPath = rootPath + "/text/textGloveVector.npy"  # 训练完成的文本嵌入矩阵

# output path
saveImageArrayDir = rootPath + "/image/imageVector/"  # 生成后的向量
saveModelWightsDir = rootPath + "/outputs/"  # 输出


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def getScore(y_true, y_pred, threshold=0.5):
    """
    :param y_true:
    :param y_pred:
    :param threshold: 阈值
    :return:
    """
    # y_true = y_true.flatten()
    # y_pred = (y_pred.flatten() > threshold).type(torch.float32)
    loss = log_loss(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_true, y_pred)  # 预测值是概率
    y_pred = (y_pred > threshold).type(torch.float32)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, pre, rec, f1, auc, loss


def getResNet50(num_classes=1000):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(block=Residual, block_num=[3, 4, 6, 3], num_classes=num_classes)


def Load_ResNet50(num_classes=1000):
    device = try_gpu()
    model_weight_path = modelWightsDir + "resnet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net = getResNet50(num_classes)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    return net


def getResNet101(num_classes=1000):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Residual, [3, 4, 23, 3], num_classes=num_classes)


def Load_ResNet101(num_classes=1000):
    device = try_gpu()
    model_weight_path = modelWightsDir + "resnet101.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net = getResNet101(num_classes)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    return net


def saveArray(array, name):
    array = array.unsqueeze(0).detach().numpy()
    numpy.save(saveImageArrayDir + name, array)


def merge_datasets(dataset, sub_dataset):
    """
    :param dataset:
    :param sub_dataset:
    :return: 合并后的dataSet
    """
    dataset.classes.extend(sub_dataset.classes)
    dataset.classes = sorted(list(set(dataset.classes)))
    dataset.class_to_idx.update(sub_dataset.class_to_idx)
    dataset.samples.extend(sub_dataset.samples)
    dataset.targets.extend(sub_dataset.targets)


def generateImageVecFiles(readImagesDirs, imageSize=480, inChannel=3, batchSize=16, blockNum=196, kernelSize=64,
                          stride=32,
                          outputSize=2048, device="gpu"):
    """ 生成图片区域向量
    :param device:
    :param imageSize: 图像统一调整为多少
    :param inChannel: 输入通道
    :param batchSize:
    :param blockNum: 一个图片分为多少个区域
    :param kernelSize: 每个区域多大
    :param stride: 步长
    :param outputSize: 输出的向量多少
    :return:
    """
    extractImageFeature = ImageRegionNet(net=getResNet50(), block_num=blockNum,
                                         kernel_size=kernelSize, stride=stride,
                                         output_size=outputSize, in_channel=inChannel)
    extractImageFeature.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(imageSize, imageSize))
    ])
    dataSet = torchvision.datasets.ImageFolder(readImagesDirs, transform=transform)
    dataIter = torch.utils.data.DataLoader(dataSet, batch_size=batchSize, shuffle=False)
    # num_workers=1)
    count = 0
    with torch.no_grad():
        if device == "gpu":
            extractImageFeature.to(try_gpu())
        for (X, y), i in zip(dataIter, tqdm(range(len(dataIter)))):
            if device == "gpu":
                X = X.cuda().type(torch.float32)
            if os.path.exists(saveImageArrayDir + dataSet.classes[int(y[0])] + ".npy"):
                continue
            batchTensor = extractImageFeature.forward(X).to(torch.device('cpu')).squeeze().numpy()
            torch.cuda.empty_cache()
            for image, index in zip(batchTensor, y):
                if os.path.exists(saveImageArrayDir + dataSet.classes[index] + ".npy"):
                    continue
                count += 1
                numpy.save(saveImageArrayDir + dataSet.classes[index], image)
    return len(dataSet)


def modelScoresVision(writer, scoresValues, scoresNames, lrValues=None):
    """
    :param lrValues:
    :param scoresNames:
    :param writer: Tensoboard
    :param scoresValues:  epochs * 评估参数个数 * 数据集字典
    :return:
    """
    if lrValues is not None:
        for batch in range(len(lrValues)):
            writer.add_scalar(tag="train_lr", scalar_value=lrValues[batch], global_step=batch)

    if scoresValues is None:
        return None
    for epoch in range(len(scoresValues) - 1):
        for i in range(len(scoresNames)):
            mapDict = {
                "train": scoresValues[epoch][i][0],
                "test": scoresValues[epoch][i][1],
                "valid": scoresValues[epoch][i][2],
            }
            writer.add_scalars(main_tag=scoresNames[i], tag_scalar_dict=mapDict, global_step=epoch)


def trainVision(writer, scoresNames, scoresValues, type, step):
    for i in range(len(scoresNames)):
        writer.add_scalars(main_tag=scoresNames[i], tag_scalar_dict={type: scoresValues[i]}, global_step=step)


def flitIllWord(line):
    """
    :param line: 数据集text文本
    :return: 数据集text list
    """
    try:
        line = eval(line)
    except:
        print(line)
    count = [1 if words in line[1] else 0 for words in illWords]
    if sum(count) != 0:
        return False
    else:
        return True


def selectErrorFileNames(main, dataType=DATASET.TEST):
    """
    :param main: Main类
    :param dataType: 数据集类别
    :return: 错误样本的id list
    """
    net = main.net
    fileNames = []
    net.eval()
    with torch.no_grad():
        for X, y, ids in main.loadData(dataType=dataType):
            y.to(torch.device(next(net.parameters()).device))
            yPred = net.forward(X).to(torch.device("cpu"))
            yPred = (yPred >= 0.5).type(torch.int)
            ids = numpy.array(ids).flatten()
            fileName = list(ids[((yPred >= 0.5).type(torch.int).flatten() != y.flatten()).numpy().flatten()])
            fileNames = fileNames + fileName
            torch.cuda.empty_cache()
    return fileNames


def preMain(sqLen=75, batchSize=64, alpha=0.7, rate=0.2):
    """
    :param sqLen:
    :param batchSize:
    :param alpha:保留源增强数据的比率
    :param rate:数据增强比例
    :return:
    """
    if input("是否重新生成数据集(Y/N):") != "Y":
        return
    dataAugment = DataAugment(seqLen=sqLen, batchSize=batchSize, imgDir=readImagesDir, textPath=textPrefix + "text.txt",
                              image2ClassPath=classPrefix + "image2classBefore.py3")
    dataList = dataAugment.start(rate=rate, imgSaveDir=imagePrefix + "/augmentImages/", image2ClassSaveDir=classPrefix,
                                 alpha=alpha)
    with open(textPrefix + "/text.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            dataList.append(line)
    trainText, dataList, _, _ = train_test_split(dataList, [0] * len(dataList), train_size=0.8, shuffle=True,
                                                 random_state=0)
    textText, validText, _, _ = train_test_split(dataList, [0] * len(dataList), train_size=0.5, shuffle=True,
                                                 random_state=0)
    with open(textPrefix + "train_text", "w+", encoding="utf-8") as file:
        for line in trainText:
            file.write("\n" + line)
    with open(textPrefix + "test_text", "w+", encoding="utf-8") as file:
        for line in textText:
            file.write("\n" + line)
    with open(textPrefix + "valid_text", "w+", encoding="utf-8") as file:
        for line in validText:
            file.write("\n" + line)
    if os.path.exists(saveImageArrayDir):
        shutil.rmtree(saveImageArrayDir)
    os.mkdir(saveImageArrayDir)
    count = generateImageVecFiles(readImagesDirs=imagePrefix + "/augmentImages/")
    count += generateImageVecFiles(readImagesDirs=readImagesDir)
    print("已生成特征向量数目：", count)


def transLogs(logsPath=time.strftime("%Y-%m-%d", time.localtime())):
    with open(saveModelWightsDir + logsPath + "/logs/2023-05-18", "rb") as f:
        logsDict, lrLogs = pickle.load(f)
    epochs = list(range(len(logsDict)))
    scores = {}
    for i in scoreNames:
        scores[i] = []
    for epoch in epochs:
        epoch = logsDict[epoch]
        for i in range(len(scoreNames)):
            scores[scoreNames[i]].append(epoch[i])
    return lrLogs, epochs, scores


from scipy.interpolate import make_interp_spline


def displayScore(logsPath=time.strftime("%Y-%m-%d", time.localtime())):
    lrLogs, epochs, scores = transLogs(logsPath)
    fig, axs = plt.subplots(row, col, figsize=(24, 16), sharex="col")
    colors = ["orange", "blue", "turquoise"]
    labels = ["train", "test", "valid"]
    axs = axs.flatten()
    for i in range(len(scores)):
        ax = axs[i]
        data = numpy.array(scores[scoreNames[i]])
        for j in range(len(labels)):
            epochsNew = numpy.linspace(min(epochs), max(epochs), 300)
            ax.plot(
                epochsNew,
                make_interp_spline(epochs, data[:, j], k=2)(epochsNew),  # 平滑处理
                "-",
                color=colors[j],
                label=labels[j],
                linewidth=3
            )
        if scoreNames[i] != "loss":
            ax.set_ylim(0.6, 1.05)
        ax.set_xlim(0, len(epochs))
        ax.legend(loc="upper left", prop={'size': 16})
        ax.set_xlabel("Epoch")
        ax.set_title(scoreNames[i])
    plt.show()
    # fig, axs = plt.subplots(1, 1, sharex="col")
    # axs.plot(
    #     list(range(len(lrLogs))),
    #     lrLogs,
    #     color="orange",
    #     label="lr"
    # )
    # axs.legend(loc="upper left")
    # axs.set_xlabel("Epoch")
    # axs.set_title("learning rate")
    # plt.show()


if __name__ == "__main__":
    displayScore("2023-05-18")

# def visualizeGridAttention(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
#                                 save_original_image=False, quality=200):
#     """
#     img_path:   image file path to load
#     save_path:  image file path to save
#     attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
#     ratio:  scaling factor to scale the output h and w
#     cmap:  attention style, default: "jet"
#     quality:  saved image quality
#     """
#     print("load image from: ", img_path)
#     img = Image.open(img_path, mode='r')
#     img_h, img_w = img.size[0], img.size[1]
#     plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
#
#     # scale the image
#     img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
#     img = img.resize((img_h, img_w))
#     plt.imshow(img, alpha=1)
#     plt.axis('off')
#
#     # normalize the attention map
#     mask = cv2.resize(attention_mask, (img_h, img_w))
#     normed_mask = mask / mask.max()
#     normed_mask = (normed_mask * 255).astype('uint8')
#     plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
#
#     if save_image:
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#         img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
#         img_with_attention_save_path = os.path.join(save_path, img_name)
#         plt.axis('off')
#         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#         plt.margins(0, 0)
#         plt.savefig(img_with_attention_save_path, dpi=quality)
#
#     if save_original_image:
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#         img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
#         original_image_save_path = os.path.join(save_path, img_name)
#         img.save(original_image_save_path, quality=quality)
