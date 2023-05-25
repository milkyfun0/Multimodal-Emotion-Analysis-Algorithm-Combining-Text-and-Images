#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 10:39
# @Author  : CaoQixuan
# @File    : 废弃代码.py
# @Description :

# self.ATT_attribute1_1, self.ATT_attribute2_1 = nn.Linear(
#     self.imageFeature.defaultFeatureSize + self.extractFeature.embSize,
#     self.imageFeature.defaultFeatureSize // 2 + self.extractFeature.embSize // 2), nn.Linear(
#     self.imageFeature.defaultFeatureSize // 2 + self.extractFeature.embSize // 2, 1)
# self.ATT_attribute1_2, self.ATT_attribute2_2 = nn.Linear(
#     self.textFeature.nHidden * 2 + self.extractFeature.embSize,
#     self.textFeature.nHidden * 2 // 2 + self.extractFeature.embSize // 2), nn.Linear(
#     self.textFeature.nHidden * 2 // 2 + self.extractFeature.embSize // 2, 1)
# self.ATT_attribute1_3, self.ATT_attribute2_3 = nn.Linear(self.extractFeature.embSize * 2,
#                                                          self.extractFeature.embSize), nn.Linear(
#     self.extractFeature.embSize, 1)
#
# self.ATT_img1_1, self.ATT_img2_1 = nn.Linear(
#     self.imageFeature.defaultFeatureSize + self.textFeature.nHidden * 2,
#     self.imageFeature.defaultFeatureSize // 2 + self.textFeature.nHidden), nn.Linear(
#     self.imageFeature.defaultFeatureSize // 2 + self.textFeature.nHidden, 1)
# self.ATT_img1_2, self.ATT_img2_2 = nn.Linear(self.imageFeature.defaultFeatureSize + self.extractFeature.embSize,
#                                              self.imageFeature.defaultFeatureSize // 2 + self.extractFeature.embSize // 2), nn.Linear(
#     self.imageFeature.defaultFeatureSize // 2 + self.extractFeature.embSize // 2, 1)
# self.ATT_img1_3, self.ATT_img2_3 = nn.Linear(self.imageFeature.defaultFeatureSize * 2,
#                                              self.imageFeature.defaultFeatureSize), nn.Linear(
#     self.imageFeature.defaultFeatureSize, 1)
#
# self.ATT_text1_1, self.ATT_text2_1 = nn.Linear(
#     self.imageFeature.defaultFeatureSize + self.textFeature.nHidden * 2,
#     self.imageFeature.defaultFeatureSize // 2 + self.textFeature.nHidden), nn.Linear(
#     self.imageFeature.defaultFeatureSize // 2 + self.textFeature.nHidden, 1)
# self.ATT_text1_2, self.ATT_text2_2 = nn.Linear(self.textFeature.nHidden * 2, self.extractFeature.embSize,
#                                                self.textFeature.nHidden + self.extractFeature.embSize // 2), nn.Linear(
#     self.textFeature.nHidden + self.extractFeature.embSize // 2, 1)
# self.ATT_text1_3, self.ATT_text2_3 = nn.Linear(self.textFeature.nHidden * 4,
#                                                self.textFeature.nHidden * 2), nn.Linear(
#     self.textFeature.nHidden * 2, 1)


# self.extractFeatureATT_Extract = Attention.AdditiveAttention(self.extractFeature.embSize,
#                                                              self.extractFeature.embSize,
#                                                              self.extractFeature.embSize // 2)
# self.extractFeatureATT_Image = Attention.AdditiveAttention(self.imageFeature.defaultFeatureSize,
#                                                            self.extractFeature.embSize,
#                                                            self.extractFeature.embSize // 2)
# self.extractFeatureATT_Extract = Attention.AdditiveAttention(self.textFeature.embSize,
#                                                              self.extractFeature.embSize,
#                                                              self.extractFeature.embSize // 2)
# self.imageFeatureATT_Extract = Attention.AdditiveAttention(self.extractFeature.embSize,
#                                                            self.imageFeature.defaultFeatureSize,
#                                                            self.imageFeature.defaultFeatureSize // 2)
# self.imageFeatureATT_Image = Attention.AdditiveAttention(self.imageFeature.defaultFeatureSize,
#                                                          self.imageFeature.defaultFeatureSize,
#                                                          self.imageFeature.defaultFeatureSize // 2)
# self.imageFeatureATT_Text = Attention.AdditiveAttention(self.textFeature.embSize,
#                                                         self.imageFeature.defaultFeatureSize,
#                                                         self.imageFeature.defaultFeatureSize // 2)
# self.textFeatureATT_Extract = Attention.AdditiveAttention(self.extractFeature.embSize,
#                                                           self.extractFeature.embSize,
#                                                           self.extractFeature.embSize // 2)


# def train_epoch(self):
#     for X, y in self.trainIter:
#         y_hat = self.net(X)  # cpu
#         y_hat = torch.cat((1 - y_hat, y_hat), dim=1)  # 应该没有弄反 20230505
#         l = self.loss(y_hat, y)  # cpu
#         if not isinstance(self.updater, torch.optim.Optimizer):
#             raise AttributeError
#         self.updater.zero_grad()
#         l.backward()
#         nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.maxClipping, norm_type=self.normType)
#         self.lr_scheduler.step(l)
#
#
# def test(self, dataType=DATASET.TEST, num=2000):
#     if isinstance(self.net, torch.nn.Module):
#         self.net.eval()
#     with torch.no_grad():
#         testData = self.loadData(dataType=dataType)
#         yPred, yTrue = [], []
#         for X, y, in testData:
#             if (dataType == DATASET.TRAIN) and (count > num):
#                 break
#             self.XExample = X
#             y_pred = self.net(X)
#             count += y_pred.shape[0]
#             yPred.append(y_pred)
#             yTrue.append(yTrue)
#     yPred = torch.cat(yPred, dim=0)
#     yTrue = (numpy.array(yTrue))
#     print(yTrue.shape)
#     return Function.getScore(y_pred=yPred, y_true=yTrue)
#
#
# def train(self):
#     maxF1 = 0  # 以F1score为指标
#     patience = self.maxPatience  # 当前的容忍度
#     if isinstance(self.net, torch.nn.Module):
#         self.net.train()
#     for epoch in range(self.maxEpoch):
#         self.train_epoch()
#         if epoch % self.displayStep:
#             # acc, pre, rec, f1, auc, loss # 元组内的顺序
#             trainScores = self.test(DATASET.TRAIN)
#             textScores = self.test(DATASET.TEST)
#             validScores = self.test(DATASET.VALID)
#             self.representationScores[epoch] = tuple(zip(trainScores, textScores, validScores))
#             end = time.time()
#             print("epoch:{}, acc:{:.3f}, rec:{.3f}, pre:{:.3f}, f1:{:.3f}, acu:{:.3f},"
#                   "loss{:.2f}, total cost:{:2.f} min".format(epoch, *textScores, (end - start) / 60))
#             if textScores[3] > maxF1 + 1e-3:
#                 maxF1, patience = textScores[3], self.maxPatience
#             else:
#                 patience -= 1
#                 if patience == 0:
#                     break

# for name, parameters in main.net.named_parameters():
#     print(name, ':', parameters.size())
# print(main.net)

# scores = main.test()
# print(scores)
# reNet = Function.getResNet50()
# print(reNet)

# train_bf = train_after
# train_after = list(main.net.named_parameters())[-1][1].grad
# print(list(main.net.named_parameters())[15][0], train_bf == train_after)
# del X, y, y_hat

# if __name__ == "__main__":
# classesPrefix = "../extract/"  # 每个图片的物品类别 ["id", "class_name" * 5]
# textPrefix = "../text/"  # 图片对应的文本 ["id", "text", "is_sarcasm"]
# imagePrefix = "../imageVector/"  # 图片的对应区域向量 id.npy
# wordsPrefix = "../words/"  # 词表
# extractWordsPrefix = "../ExtractWords/"  # 类名对应的编号和GLove向量
# train_data = MyDataSet(
#     seqLen=10,
#     classPrefix=extractWordsPrefix,
#     imageVectorPrefix=r"D:\Code\PyCharm\data-of-multimodal-sarcasm-detection\imageVector2\\",
#     textPath=textPrefix,
#     wordVocabDir=wordsPrefix
# )
# train_loader = DataLoader(dataset=train_data, batch_size=5, shuffle=True)
# tokenizer = AutoTokenizer.from_pretrained(modelWightsDir + "bert-base-cased")
# encoded_input = tokenizer(["12222", "1222222222222222222"], return_tensors='pt', padding="max_length",
#                           truncation=True,
#                           max_length=3)
# print(encoded_input["input_ids"].shape)

#
# summaryWriter = SummaryWriter(log_dir=r"C:\Users\alice\Desktop\results\modelwights\2023-05-09\runs\\")
# with open(r"C:\Users\alice\Desktop\results\modelwights\2023-05-09\logs\2023-05-09", 'rb') as f:
#     representationScores, lrRecord = pickle.load(f)
#     modelScoresVision(summaryWriter, scoresValues=representationScores,
#                       scoresNames=["acc", "pre", "rec", "f1", "auc", "loss"], lrValues=lrRecord)

# import  os
#
# class DataSet(data.Dataset):
#     """
#     自定义的数据集参数,用于提取图片的特征向量
#     """
#
#     def __init__(self, img_dir, resize):
#         super(DataSet, self).__init__()
#         self.img_paths = glob('{:s}/*'.format(img_dir))
#         self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])
#
#     def __getitem__(self, item):
#         img = Image.open(self.img_paths[item]).convert('RGB')
#         img = self.transform(img)
#
#         return img, self.img_paths[item]
#
#     def __len__(self):
#         return len(self.img_paths)
#
#
# def ProcessPreImages(img_dir, resize, save_dir):
#     """
#     :param img_dir:
#     :param resize: 改为需要的大小
#     :param save_dir:
#     :return:
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_dir', type=str, default=img_dir)
#     parser.add_argument('--resize', type=int, default=resize)
#     parser.add_argument('--save_dir', type=str, default=save_dir)
#     args = parser.parse_args()
#     if not os.path.exists(args.save_dir):
#         os.mkdir(args.save_dir)
#     else:
#         if len(os.listdir(img_dir)) >= 1:  # 说明已经有文件了 - 默认已经处理完了图片
#             return None
#
#     dataset = DataSet(args.img_dir, args.resize)
#     print('dataset:', len(dataset))
#     count = 0
#     start = time.time()
#     for i in range(len(dataset)):
#         img, path = dataset[i]
#         path = os.path.basename(path)
#         if count % 1000 == 0:
#             print('Processing: ', count, " files")
#         count += 1
#         if not os.path.exists(args.save_dir + "/{:s}".format(path[0:-4])):  # 生成transformer要求的数据集格式
#             os.mkdir(args.save_dir + "/{:s}".format(path[0:-4]))
#         imageio.imwrite(args.save_dir + '/{:s}/{:s}'.format(path[0:-4], path), img)
#     end = time.time()
#     print("finished total cost: {:.2f} min".format((end - start) / 60))
#
#
# def getDatasetIter(img_dir, batch_size, shuffle=True, num_workers=6):
#     """
#     :param img_dir:
#     :param batch_size: 批量大小
#     :param shuffle: 是否随机
#     :param num_workers: 使用的线程数
#     :return: 数据集, 类别名称
#     """
#     transform = transforms.ToTensor()
#     train_data = torchvision.datasets.ImageFolder(img_dir, transform=transform)
#     trainIter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
#                                              num_workers=num_workers)
#     return trainIter, train_data.classes
#
#
# def generateImageVecFiles(imageSize=480, inChannel=3, batchSize=4, blockNum=196, kernelSize=64, stride=32,
#                           outputSize=2048):
#     """
#     :param imageSize: 图像统一调整为多少
#     :param inChannel: 输入通道
#     :param batchSize:
#     :param blockNum: 一个图片分为多少个区域
#     :param kernelSize: 每个区域多大
#     :param stride: 步长
#     :param outputSize: 输出的向量多少
#     :return:
#     """
#     net = getResNet50().to(device=try_gpu())
#
#     if not os.path.exists(saveImageArrayDir):
#         os.mkdir(saveImageArrayDir)
#     else:
#         if len(os.listdir(saveImageArrayDir)) >= 1:  # 说明已经有文件了 - 默认已经得到了向量
#             return None
#     if not os.path.exists(saveImagesDir):
#         os.mkdir(saveImagesDir)
#
#     ProcessPreImages(readImagesDirs, imageSize, saveImagesDir)
#     preVectorIter, classes = getDatasetIter(saveImagesDir, batch_size=batchSize, shuffle=False)
#     extractImageFeature = ImageFeature(net=net, block_num=blockNum,
#                                        kernel_size=kernelSize, stride=stride,
#                                        output_size=outputSize, in_channel=inChannel)
#
#     def saveArray(array, index):
#         array = array.unsqueeze(0).detach().numpy()
#         numpy.save(saveImageArrayDir + classes[int(index)], array)
#
#     count = 0
#     net.eval()
#     start = time.time()
#     with torch.no_grad():
#         for X, y in preVectorIter:
#             end = time.time()
#             if count % (batchSize * 50) == 0:
#                 print("have got {} image vectors, total cost:{:.2f} min".format(count, (end - start) / 60))
#             count += batchSize
#             if os.path.exists(saveImageArrayDir + classes[int(y[0])] + ".npy"):
#                 continue
#             batch_tensor = extractImageFeature.forward(X.type(torch.float32).cuda()).to(torch.device('cpu'))
#             torch.cuda.empty_cache()
#             [saveArray(data, index) for data, index in zip(batch_tensor, y)]  # 加速
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 17:09
# @Author  : CaoQixuan
# @File    : VisualAttention.py
# @Description :

# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
#
# matplotlib.use('Agg')
#
#
# def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
#                                 save_original_image=False, quality=200):
#     """
#     img_path:   image file path to load
# #     save_path:  image file path to save
# #     attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
# #     ratio:  scaling factor to scale the output h and w
# #     cmap:  attention style, default: "jet"
# #     quality:  saved image quality
# #     """
# #     print("load image from: ", img_path)
# #     img = Image.open(img_path, mode='r')
# #     img_h, img_w = img.size[0], img.size[1]
# #     plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
# #
# #     # scale the image
# #     img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
# #     img = img.resize((img_h, img_w))
# #     plt.imshow(img, alpha=1)
# #     plt.axis('off')
# #
# #     # normalize the attention map
# #     mask = cv2.resize(attention_mask, (img_h, img_w))
# #     normed_mask = mask / mask.max()
# #     normed_mask = (normed_mask * 255).astype('uint8')
# #     plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
# #
# #     if save_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
# #         img_with_attention_save_path = os.path.join(save_path, img_name)
# #         plt.axis('off')
# #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# #         plt.margins(0, 0)
# #         plt.savefig(img_with_attention_save_path, dpi=quality)
# #
# #     if save_original_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
# #         original_image_save_path = os.path.join(save_path, img_name)
# #         img.save(original_image_save_path, quality=quality)
#
#
# def selectErrorFileNames(modelWeightDir=time.strftime("%Y-%m-%d", time.localtime()),
#                          dataType=DATASET.TEST, device="gpu"):
#     main = Main(device="cpu")
#     main.loadNet(modelWeightDir)
#     net = main.net
#     fileNames = numpy.array([])
#     net.eval()
#     # with torch.no_grad():
#     #     for X, y, ids in main.loadData(dataType=dataType):
#     #         yPred = net.forward(X).to(torch.device("cpu"))
#     #         ids = numpy.array(ids).flatten()
#     #         fileNames = numpy.append(fileNames,
#     #                                  ids[((yPred >= 0.5).type(torch.int).flatten() != y.flatten()).numpy().flatten()])
#     #         break
#     # return fileNames
#     with torch.no_grad():
#         numpy.save("../weight", net.state_dict()['imageFeatureATT.addATT_1.W_k.weight'].numpy())
#
#
# if __name__ == "__main__":
#     selectErrorFileNames()
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 17:09
# @Author  : CaoQixuan
# @File    : VisualAttention.py
# @Description :

# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
#
# matplotlib.use('Agg')
#
#
# def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
#                                 save_original_image=False, quality=200):
#     """
#     img_path:   image file path to load
# #     save_path:  image file path to save
# #     attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
# #     ratio:  scaling factor to scale the output h and w
# #     cmap:  attention style, default: "jet"
# #     quality:  saved image quality
# #     """
# #     print("load image from: ", img_path)
# #     img = Image.open(img_path, mode='r')
# #     img_h, img_w = img.size[0], img.size[1]
# #     plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
# #
# #     # scale the image
# #     img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
# #     img = img.resize((img_h, img_w))
# #     plt.imshow(img, alpha=1)
# #     plt.axis('off')
# #
# #     # normalize the attention map
# #     mask = cv2.resize(attention_mask, (img_h, img_w))
# #     normed_mask = mask / mask.max()
# #     normed_mask = (normed_mask * 255).astype('uint8')
# #     plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
# #
# #     if save_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
# #         img_with_attention_save_path = os.path.join(save_path, img_name)
# #         plt.axis('off')
# #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# #         plt.margins(0, 0)
# #         plt.savefig(img_with_attention_save_path, dpi=quality)
# #
# #     if save_original_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
# #         original_image_save_path = os.path.join(save_path, img_name)
# #         img.save(original_image_save_path, quality=quality)
#
#
# def selectErrorFileNames(modelWeightDir=time.strftime("%Y-%m-%d", time.localtime()),
#                          dataType=DATASET.TEST, device="gpu"):
#     main = Main(device="cpu")
#     main.loadNet(modelWeightDir)
#     net = main.net
#     fileNames = numpy.array([])
#     net.eval()
#     # with torch.no_grad():
#     #     for X, y, ids in main.loadData(dataType=dataType):
#     #         yPred = net.forward(X).to(torch.device("cpu"))
#     #         ids = numpy.array(ids).flatten()
#     #         fileNames = numpy.append(fileNames,
#     #                                  ids[((yPred >= 0.5).type(torch.int).flatten() != y.flatten()).numpy().flatten()])
#     #         break
#     # return fileNames
#     with torch.no_grad():
#         numpy.save("../weight", net.state_dict()['imageFeatureATT.addATT_1.W_k.weight'].numpy())
#
#
# if __name__ == "__main__":
#     selectErrorFileNames()
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 17:09
# @Author  : CaoQixuan
# @File    : VisualAttention.py
# @Description :
# import numpy
# import torch
# from DATASET import DATASET
# from codes.Main import Main
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
#
# matplotlib.use('Agg')
#
#
# def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
#                                 save_original_image=False, quality=200):
#     """
#     img_path:   image file path to load
# #     save_path:  image file path to save
# #     attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
# #     ratio:  scaling factor to scale the output h and w
# #     cmap:  attention style, default: "jet"
# #     quality:  saved image quality
# #     """
# #     print("load image from: ", img_path)
# #     img = Image.open(img_path, mode='r')
# #     img_h, img_w = img.size[0], img.size[1]
# #     plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
# #
# #     # scale the image
# #     img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
# #     img = img.resize((img_h, img_w))
# #     plt.imshow(img, alpha=1)
# #     plt.axis('off')
# #
# #     # normalize the attention map
# #     mask = cv2.resize(attention_mask, (img_h, img_w))
# #     normed_mask = mask / mask.max()
# #     normed_mask = (normed_mask * 255).astype('uint8')
# #     plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
# #
# #     if save_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
# #         img_with_attention_save_path = os.path.join(save_path, img_name)
# #         plt.axis('off')
# #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# #         plt.margins(0, 0)
# #         plt.savefig(img_with_attention_save_path, dpi=quality)
# #
# #     if save_original_image:
# #         if not os.path.exists(save_path):
# #             os.mkdir(save_path)
# #         img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
# #         original_image_save_path = os.path.join(save_path, img_name)
# #         img.save(original_image_save_path, quality=quality)
#
#
# def selectErrorFileNames(modelWeightDir=time.strftime("%Y-%m-%d", time.localtime()),
#                          dataType=DATASET.TEST, device="gpu"):
#     main = Main(device="cpu")
#     main.loadNet(modelWeightDir)
#     net = main.net
#     fileNames = numpy.array([])
#     net.eval()
#     # with torch.no_grad():
#     #     for X, y, ids in main.loadData(dataType=dataType):
#     #         yPred = net.forward(X).to(torch.device("cpu"))
#     #         ids = numpy.array(ids).flatten()
#     #         fileNames = numpy.append(fileNames,
#     #                                  ids[((yPred >= 0.5).type(torch.int).flatten() != y.flatten()).numpy().flatten()])
#     #         break
#     # return fileNames
#     with torch.no_grad():
#         numpy.save("../weight", net.state_dict()['imageFeatureATT.addATT_1.W_k.weight'].numpy())
#
#
# if __name__ == "__main__":
#     selectErrorFileNames()
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/17 10:01
# @Author  : CaoQixuan
# @File    : text.py
# @Description :
# import numpy
# #
# # main = Main()
# # model = main.net
# # # for name in main.net.state_dict():
# # #     print(name)
# numpy.save("../tensor.pt", model.state_dict()['imageFeatureATT.addATT_1.W_k.weight'].numpy())
# import pickle
#
# #
# with open("../text/textVocab.py3", "rb") as f:
#     image2Class = pickle.load(f)
#
# print(image2Class.keys())
# # # imageNames = os.listdir(r"D:\Code\PyCharm\Paper\dataset_images\\")
# # # print(imageNames)
# # # with open("../text/text.txt", "r", encoding="utf-8") as file:
# # #     with open("../text/text", "w+", encoding="utf-8") as f:
# # #         for line in file:
# # #             l = eval(line)
# # #             if l[0] not in image2Class:
# # #                 print("123")
# # #                 continue
# # #             if l[0] + ".jpg" not in imageNames:
# # #                 print("123")
# # #                 continue
# # #             f.write(str(l) + "\n")
# #
# # TT = {}
# # for key in image2Class.keys():
# #     TT[str(key)] = image2Class[key]
# # with open("../text/textVocab.py3", "wb+") as f:
# #     pickle.dump(TT, f)
# 下面两个学习率衰减用法不一样
# self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=self.updater,
#     mode="min",  # 增加/ 减小
#     patience=20,  # loos/acc 不再减小（或增大）的累计次数后改变学习率；
#     verbose=False,  # 是否可视
#     min_lr=1e-7,  # 最小的学习率
#     cooldown=10,  # 更新后冷静期
#     eps=1e-3  # If the difference between new and old lr is smaller than eps, the update is ignored
# )  # 在发现loss不再降低或者acc不再提高之后，降低学习率，这里用于批量的，所以呢，循环论数很多， 大约是 20K / batch_size

# print(len(dataSet))
# for i in range(len(readImagesDirs) - 1):
#     subSet = torchvision.datasets.ImageFolder(readImagesDirs[i + 1], transform=transform)
#     merge_datasets(dataSet, subSet)
# print(len(dataSet))

# if __name__ == "__main__":
#     bert = TextFeature_Bert(20, 20, 0.1)
#     print(bert.forward(["2222", "222222"])[0].shape)
