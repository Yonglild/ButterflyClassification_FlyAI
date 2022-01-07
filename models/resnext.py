import torch
import torch.nn as nn
import torchvision

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper


PATH = {
    'resnext50' : 'https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth',
    'resnext101': 'https://www.flyai.com/m/resnext101_32x8d-8ba56ff5.pth',
}

in_features = {
    'resnext50' : 2048,
    'resnext101': 2048,
}

def net(model_name, NUM_CLASS, is_pretrained=True):

    model_path = remote_helper.get_remote_date(PATH[model_name])

    if model_name == 'resnext50':
        model = torchvision.models.resnext50_32x4d(pretrained=False)
    elif model_name == 'resnext101':
        model = torchvision.models.resnext101_32x8d(pretrained=False)
    else:
        print('Error model name')

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