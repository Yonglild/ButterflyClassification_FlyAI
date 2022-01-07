# -*- coding: utf-8 -*
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

from flyai.framework import FlyAI
from torch.autograd import Variable
from torchvision import transforms

from path import MODEL_PATH, DATA_PATH
from PIL import ImageFile
from utils import auto_augment
from utils.data_wrap import get_label_indx

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTL_TIMES = 1

class Prediction(FlyAI):
    def load_model(self):
        # model1 = torch.load(MODEL_PATH + '/' + "best1.pth")
        # model1.eval()
        # self.model1 = model1.to(device)

        model2 = torch.load(MODEL_PATH + '/' + "best2.pth")
        model2.eval()
        self.model2 = model2.to(device)
        #
        # model3 = torch.load(MODEL_PATH + '/' + "best3.pth")
        # model3.eval()
        # self.model3 = model3.to(device)

        self.labels, _ = get_label_indx()

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')

        test_trainsform = transforms.Compose([
            transforms.Resize((252, 336)),
            transforms.CenterCrop((168, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            # transforms.RandomPerspective(distortion_scale=0.2),
            # auto_augment.AutoAugment(dataset='IMAGENET'),
            # auto_augment.AutoAugment(dataset='CIFAR'),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        output = 0
        for i in range(TTL_TIMES):
            tensor = test_trainsform(img)
            tensor = torch.unsqueeze(tensor, dim=0).float()
            tensor = tensor.to(device)
            # output1 = self.model1(tensor)
            output2 = self.model2(tensor)
            # output3 = self.model3(tensor)
            output += output2 #+ output3

        pred = output.max(1, keepdim=True)[1]
        pred = self.labels[pred]
        # print(pred)
        return {"label": pred}

if __name__ == '__main__':
    import glob
    Predict = Prediction()
    Predict.load_model()
    globlist = glob.glob('/home/wyl/AI_competition/ButterflyClassification_FlyAI/data/input/ButterflyClassification/image/*jpg')
    for imgpath in globlist:
        print(Predict.predict(imgpath))