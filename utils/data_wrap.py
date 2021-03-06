from __future__ import print_function, absolute_import

import os
import pandas as pd
import random
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
import numpy as np
from path import DATA_PATH, DATA_ID


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def get_label_indx():
    df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
    label_list = df['label'].values
    result = []
    for i in label_list:
        if i not in result:
            result.append(i)
    # print('len(tmp_label):', len(result))
    # print(result)

    num_count = {}
    for i in label_list:
        if i not in num_count:
            num_count[i] = 1
        else:
            num_count[i] += 1
    # num_count_sort = sorted(num_count.items(), key=lambda i:i[1], reverse=True)
    # print('num_count:', num_count_sort)
    return result, num_count

class ImageData(Dataset):
    def __init__(self, path_list, label_list, transform):
        self.dataset = path_list
        self.label = label_list
        self.transform = transform
        print('len(dataset):', len(self.dataset))
        # self.dict = {}
        # self.dict = dict.fromkeys(self.label, np.arange(0, 2000))
        # self.tmp_label = [self.tmp_label.append(i) for i in self.label if not i in self.label]
        self.tmp_label, _ = get_label_indx()

    def __getitem__(self, item):
        img = self.dataset[item]
        img_label = self.label[item]
        label = self.tmp_label.index(img_label)

        img = read_image(os.path.join(DATA_PATH, DATA_ID, img))
        if img.height > img.width:
            img = img.transpose(Image.ROTATE_90)
        if item > 20223:
            degree = random.randint(-60, 60)
            # print('degree:', degree)
            img = img.rotate(degree)

        # ratio = 0.9
        # b1, a1, b2, a2 = (1-ratio)*img.width, (1-ratio)*img.height, ratio*img.width, ratio*img.height
        # img = img.crop((int(b1), int(a1), int(b2), int(a2)))

        if self.transform is not None:
            img = self.transform(img)
        # img.show()
        return img, label

    def __len__(self):
        return len(self.dataset)


class TestImageData(Dataset):
    def __init__(self, path_list, transform):

        self.dataset = path_list
        self.transform = transform

    def __getitem__(self, item):
        imgname = self.dataset[item]

        img = read_image(os.path.join(DATA_PATH, DATA_ID, imgname))

        if self.transform is not None:
            img = self.transform(img)
            
        return img, imgname

    def __len__(self):
        return len(self.dataset)