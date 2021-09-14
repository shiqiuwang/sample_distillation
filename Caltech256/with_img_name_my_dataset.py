# -*- coding: utf-8 -*-

'''
@Time    : 2021/7/26 19:44
@Author  : Qiushi Wang
@FileName: with_img_name_my_dataset.py
@Software: PyCharm
'''

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {}
for i in range(257):
    rmb_label[str(i)] = i


# rmb_label = {"0": 0, "1": 1, "2": 2, "3": 3}


class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = rmb_label
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        image_name, path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255
        # img = Image.open(path_img)  # 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return path_img, img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                # count = 0
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((img_name, path_img, int(label)))
        return data_info