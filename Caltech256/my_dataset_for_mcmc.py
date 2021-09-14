# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/18 17:15
@Author  : Qiushi Wang
@FileName: my_dataset_for_mcmc.py
@Software: PyCharm
'''

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
# rmb_label = {}
# for i in range(20):
#     rmb_label[str(i)] = i



# rmb_label = {"0": 0, "1": 1, "2": 2, "3": 3}


class RMBDatasetMCMC(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # self.label_name = rmb_label
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255
        # img = Image.open(path_img)  # 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()

        path_images = []

        with open(file=data_dir, mode='r', encoding='utf-8') as f:
            for img in f.readlines():
                path_images.append(img.strip())

        for img in path_images:
            path_image = img
            label = int(img.split("/")[4])

            data_info.append((path_image, label))

        return data_info
