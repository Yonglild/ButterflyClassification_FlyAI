import torch
import torch.nn as nn
import torchvision

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper


# need to change!!!
PATH = {
    'resnet18': 'https://www.flyai.com/m/resnet18-5c106cde.pth',
    'resnet34': 'https://www.flyai.com/m/resnet34-333f7ec4.pth',
    'resnet50': 'https://www.flyai.com/m/resnet50-19c8e357.pth',
    'resnet101': 'https://www.flyai.com/m/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://www.flyai.com/m/resnet152-b121ed2d.pth',
}

in_features = {
    'resnet18' : 512,
    'resnet34' : 512,
    'resnet50' : 2048,
    'resnet101': 2048,
    'resnet152': 2048,
}

def net(model_name, NUM_CLASS=1000, is_pretrained=True):

    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=False)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=False)
    else:
        print('Error model name')

    model_path = remote_helper.get_remote_date(PATH[model_name])

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