import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd


DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')

DATA_ID = 'ButterflyClassification'

######################## 图像尺寸分布 ###################################
hw = []
file_list = glob.glob('/home/wyl/AI_competition/ButterflyClassification_FlyAI/data/input/ButterflyClassification/image/*.jpg')
for file in file_list:
    img = cv2.imread(file)
    h, w = img.shape[:2]
    hw.append([h, w])

hw = np.array(hw)
# plt.plot(hw[:, 0], hw[:, 1], 'g')
plt.scatter(hw[:, 0], hw[:, 1], c='r')
# plt.show()
plt.savefig('hw')

######################## 图像类别分布 ####################################
def get_label_indx():
    df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
    label_list = df['label'].values
    labels = []
    for i in label_list:
        if i not in labels:
            labels.append(i)
    print('len(tmp_label):', len(labels))

    num_count = {}
    for i in label_list:
        if i not in num_count:
            num_count[i] = 1
        else:
            num_count[i] += 1
    num_count = sorted(num_count.items(), key=lambda i:i[1], reverse=True)
    return labels, num_count

labels, num_count = get_label_indx()
print(labels)
print(num_count)

plt.bar([x[0] for x in num_count], [x[1] for x in num_count])
# plt.show()
plt.savefig('bar.png')