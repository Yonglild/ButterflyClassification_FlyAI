import torch
import torch.nn as nn
import torchvision

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper

torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

PATH = {
    'resnest50' : 'https://www.flyai.com/m/resnest50-528c19ca.pth',
    'resnest101': 'https://www.flyai.com/m/resnest101-22405ba7.pth',
    'resnest200': 'https://www.flyai.com/m/resnest200-75117900.pth',
    'resnest269': 'https://www.flyai.com/m/resnest269-0cc87c48.pth',
}

in_features = {
    'resnest50' : 2048,
    'resnest101': 2048,
    'resnest200': 2048,
    'resnest269': 2048
}


def net(model_name, NUM_CLASS, is_pretrained=True):

    model_path = remote_helper.get_remote_date(PATH[model_name])
    
    model = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=False)

    if is_pretrained:
        model.load_state_dict(torch.load(model_path), strict=False)

    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features[model_name]),
        nn.Linear(in_features[model_name], NUM_CLASS*4),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(NUM_CLASS*4),
        nn.Linear(NUM_CLASS*4, NUM_CLASS)
    )

    return model