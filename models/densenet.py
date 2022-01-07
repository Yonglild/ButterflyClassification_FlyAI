import torch
import torch.nn as nn
import torchvision

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper

PATH = {
    'densenet121' : 'https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth',
    'densenet161' : 'https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth',
    'densenet169' : 'https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth',
    'densenet201' : 'https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth',
}

in_features = {
    'densenet121' : 1024,
    'densenet161' : 2208,
    'densenet169' : 1664,
    'densenet201' : 1920,
}

def net(model_name, NUM_CLASS, is_pretrained=True):

    model_path = remote_helper.get_remote_date(PATH[model_name])

    if model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=False)
    elif model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=False)
    elif model_name == 'densenet169':
        model = torchvision.models.densenet169(pretrained=False)
    elif model_name == 'densenet201':
        model = torchvision.models.densenet201(pretrained=False)
    else:
        print('Error model name')

    if is_pretrained:
        model.load_state_dict(torch.load(model_path), strict=False)

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features[model_name]),
        nn.Linear(in_features[model_name], NUM_CLASS*4),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(NUM_CLASS*4),
        nn.Linear(NUM_CLASS*4, NUM_CLASS)
    )

    return model