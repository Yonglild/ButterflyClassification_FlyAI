import torch
import torch.nn as nn
import torchvision

from efficientnet_pytorch import EfficientNet

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper



PATH = {
    'efficientnet-b0' : 'https://www.flyai.com/m/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1' : 'https://www.flyai.com/m/efficientnet-b1-f1951068.pth',
    'efficientnet-b2' : 'https://www.flyai.com/m/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3' : 'https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4' : 'https://www.flyai.com/m/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5' : 'https://www.flyai.com/m/efficientnet-b5-b6417697.pth',
    'efficientnet-b6' : 'https://www.flyai.com/m/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7' : 'https://www.flyai.com/m/efficientnet-b7-dcc49843.pth',
}

in_features = {
    'efficientnet-b0' : 1280, 
    'efficientnet-b1' : 1280, 
    'efficientnet-b2' : 1408, 
    'efficientnet-b3' : 1536, 
    'efficientnet-b4' : 1792, 
    'efficientnet-b5' : 2048, 
    'efficientnet-b6' : 2304, 
    'efficientnet-b7' : 2560, 
}


def net(model_name, NUM_CLASS, is_pretrained=True):

    model_path = remote_helper.get_remote_date(PATH[model_name])

    model = EfficientNet.from_name(model_name)

    if is_pretrained:
        model.load_state_dict(torch.load(model_path), strict=False)

    model._fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features[model_name],NUM_CLASS)
    )

    return model