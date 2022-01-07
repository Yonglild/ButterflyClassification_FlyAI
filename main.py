# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from torch.utils.data import DataLoader
from torchvision import transforms

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH
import math
import time
from PIL import Image, ImageFilter
from utils.data_wrap import ImageData
from utils import auto_augment, losses
from path import MODEL_PATH, DATA_PATH, DATA_ID, NUM_CLASS
from models import senet#, densenet#, efficientnet #, resnext, senet, resnet, resnest,
# from cnn_finetune import make_model
from models import resnet_cbam
from utils.data_wrap import  get_label_indx
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# # 项目的超参，不使用可以删除
# parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
# parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
# args = parser.parse_args()


epochs = 15
BATCH_SIZE = 32
VAL_BATCH_SIZE = 1

TRAIN_PER = 0.75

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# whether use gpu
use_gpu = torch.cuda.is_available()
if use_gpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("ButterflyClassification")

    def data_process(self):
        normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ## 冗余面积有点大
        train_transform = transforms.Compose([
            transforms.Resize((252, 336)),
            transforms.CenterCrop((168, 224)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomPerspective(distortion_scale=0.2),
            # transforms.RandomAffine(degrees=2, translate=(0.05, 0.08), scale=(0.9, 1.1)),
            # transforms.RandomRotation(5),
            # auto_augment.AutoAugment(dataset='IMAGENET'),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 1.5)),
            normal
        ])

        test_trainsform = transforms.Compose([
            transforms.Resize((252, 336)),
            transforms.CenterCrop((168, 224)),
            transforms.ToTensor(),
            normal
        ])
        return train_transform, test_trainsform

    def train_model(self, model, train_loader, val_loader, save_model_name, epochs):
        model.to(DEVICE)

        # criteration = nn.CrossEntropyLoss()
        criteration = losses.CrossEntropyLabelSmooth(NUM_CLASS)
        # criteration = losses.FocalLoss(NUM_CLASS)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        print(' optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)')
        #
        # # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs/6), eta_min=1e-5, last_epoch=-1)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 9], gamma=0.2)             # milestones=[int(epochs / 3), int(2 * epochs / 3),int(5 * epochs / 6)], gamma=0.2)

        warm_up_epochs = 4
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
                math.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)  # CosineAnnealingLR   lamda * initial_lr
        print("torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)")


        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for i, (img, label) in enumerate(train_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                output = model(img)
                loss = criteration(output, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if i % 20 == 0:
                    print(f"Epoch: {epoch}, Step {i} | {len(train_loader)}, Train_Loss {train_loss:.4f}")
                    train_loss = 0

            scheduler.step(epoch)
            print('Current Time:', time.asctime())

            if epoch % 1 == 0 or epoch == epochs - 1:
                correct = 0
                model.eval()
                for val_img, val_label in val_loader:
                    val_img, val_label = val_img.to(DEVICE), val_label.to(DEVICE)
                    val_output = model(val_img)
                    val_pred = val_output.max(1, keepdim=True)[1]
                    correct += val_pred.eq(val_label.view_as(val_pred)).sum().item()

                print(f"Epoch {epoch},  Accuracy {100 * correct / len(val_loader.dataset):.4f}%")
        torch.save(model, MODEL_PATH + '/' + save_model_name)
        print('Current model ends, saved the model.')


    def train(self):
        df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        image_path_list = df['image_path'].values
        label_list = df['label'].values
        print('Label info', np.sum(label_list))

        # split dataset
        all_size = len(image_path_list)
        train_size = int(all_size * TRAIN_PER)

        train_image_path_list = image_path_list
        train_label_list = label_list
        # train_image_path_list = image_path_list
        # train_label_list = label_list
        _, num_count = get_label_indx()
        num_count_sort = sorted(num_count.items(), key=lambda i:i[1], reverse=True)
        print('num_count1:', num_count_sort)
        print('before augment:', len(train_image_path_list))

        """"扩增数据集"""
        # train_image_path_list = np.hstack((train_image_path_list, image_path_list[:5000]))
        # train_label_list = np.hstack((train_label_list, label_list[:5000]))
        # print('after augment:', len(train_image_path_list))

        tmp_img, tmp_label = [], []
        for _ in range(4):
            for i in range(len(label_list)):
                label = label_list[i]
                img_path = image_path_list[i]
                # if num_count[label] > 400:
                #     np.delete(train_image_path_list, i)
                #     np.delete(label_list, i)
                #     num_count[label] -= 1
                if num_count[label] < 150:
                    tmp_img.append(img_path)
                    tmp_label.append(label)
                    num_count[label] += 1

        train_image_path_list = np.hstack((train_image_path_list, tmp_img))
        train_label_list = np.hstack((train_label_list, tmp_label))
        num_count_sort = sorted(num_count.items(), key=lambda i:i[1], reverse=True)
        print('num_count2:', num_count_sort)
        print('after augment:', len(train_image_path_list), len(train_label_list))


        val_image_path_list = image_path_list[train_size:]
        val_label_list = label_list[train_size:]
        print(
            'train_size: %d, val_size: %d' % (len(train_image_path_list), len(val_image_path_list)))

        # 数据预处理
        train_transform, val_trainsform = self.data_process()

        train_data = ImageData(train_image_path_list, train_label_list, train_transform)
        val_data = ImageData(val_image_path_list, val_label_list, val_trainsform)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=True)

        # model1 = senet.net('seresnext101', NUM_CLASS, is_pretrained=True)
        # print('Loading model1...')
        # self.train_model(model1, train_loader, val_loader, 'best1.pth', epochs)

        model2 = resnet_cbam.resnext101_32x8d(pretrained=True)
        model2.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, NUM_CLASS)
        )
        print('Loading model2...')
        self.train_model(model2, train_loader, val_loader, 'best2.pth', epochs)

        # model3 = senet.net('seresnext101', NUM_CLASS, is_pretrained=True)
        #
        # print('Loading model3...')
        # self.train_model(model3, train_loader, val_loader, 'best3.pth', epochs)
        #
        # model3 = efficientnet.net('efficientnet-b3', NUM_CLASS, is_pretrained=True)
        #
        # print('Loading model3...')
        # self.train_model(model3, train_loader, val_loader, 'best3.pth', epochs + 1)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()