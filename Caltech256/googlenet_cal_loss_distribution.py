# -*- coding: utf-8 -*-

'''
@Time    : 2021/7/26 21:42
@Author  : Qiushi Wang
@FileName: googlenet_new_cal_loss_distribution.py
@Software: PyCharm
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from with_img_name_my_dataset import RMBDataset
import os
import random
from common_tools import LabelSmoothingCrossEntropy

device = torch.device("cpu")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.inception_v3(num_classes=257)
    pretrained_state_dict = torch.load(path_state_dict, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_state_dict.state_dict())

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    # model.to(device)
    return model


# 参数设置
BATCH_SIZE = 64

# ============================ step 1/5 数据 ============================

# split_dir = os.path.join("..", "..", "data", "rmb_split")
split_dir = "./data/caltech_split_data"
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

path_state_dict = './pkl/finish_model_googlenet.pkl'

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((342)),  # 299 / (224/256) = 342
    transforms.CenterCrop(299),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((342)),  # 299 / (224/256) = 342
    transforms.CenterCrop(299),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
print(len(train_data))
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
print(len(valid_data))
# print("len(train_data)", len(train_data))
# print("len(train_loader)", len(train_loader))

# ============================ step 2/5 模型 ============================

alexnet_model = get_model(path_state_dict, False)
alexnet_model = alexnet_model.to(device)

# num_ftrs = alexnet_model.classifier._modules["6"].in_features
# alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)
#
# alexnet_model.to(device)
# ============================ step 3/5 损失函数 ============================
criterion = LabelSmoothingCrossEntropy(eps=0.001)
# ============================ step 4/5 优化器 ============================
# 冻结卷积层
# flag = 0
# # flag = 1
# if flag:
#     fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
#     base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
#     optimizer = optim.SGD([
#         {'params': base_params, 'lr': LR * 0.1},  # 0
#         {'params': alexnet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)
#
# else:
# optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
# train_curve = list()
# valid_curve = list()
# select_list = []

# data_idx_dict = dict()
# select_idx_list = set()
# train_idx_to_real_index = {}
train_loss_dict = dict()  # name:loss
valid_loss_dict = dict()  # name:loss
# sim_dict = dict()  # index:similarity
# label_dict = dict()  # index:label

alexnet_model.eval()
# count = 0
with torch.no_grad():
    for i, data in enumerate(train_loader):
        name, inputs, labels = data
        labels = labels.view(-1, 1)
        # if hasattr(torch.cuda, "empty_cache"):
        #     torch.cuda.empty_cache()
        inputs = inputs.to(device)
        # count += 1
        # print(count)
        labels = labels.to(device)

        for idx in range(len(inputs)):
            # sim_dict[i * BATCH_SIZE + idx] = inputs[idx].cpu().numpy().var()
            # data_idx_dict[i * BATCH_SIZE + idx] = [inputs[idx], labels[idx]]
            # train_idx_to_real_index[i * BATCH_SIZE + idx] = index[idx].cpu().item()

            outputs = alexnet_model(inputs[idx].unsqueeze_(0))
            loss = criterion(outputs, labels[idx])
            train_loss_dict[name[idx]] = loss.cpu().item()
            # label_dict[i * BATCH_SIZE + idx] = labels[idx].cpu().item()
    print("train dataset finish!!!")
    for j, data in enumerate(valid_loader):
        name, inputs, labels = data
        labels = labels.view(-1, 1)
        # if hasattr(torch.cuda, "empty_cache"):
        #     torch.cuda.empty_cache()
        inputs = inputs.to(device)
        # count += 1
        # print(count)
        labels = labels.to(device)

        for idx in range(len(inputs)):
            # sim_dict[i * BATCH_SIZE + idx] = inputs[idx].cpu().numpy().var()
            # data_idx_dict[i * BATCH_SIZE + idx] = [inputs[idx], labels[idx]]
            # train_idx_to_real_index[i * BATCH_SIZE + idx] = index[idx].cpu().item()

            outputs = alexnet_model(inputs[idx].unsqueeze_(0))
            loss = criterion(outputs, labels[idx])
            valid_loss_dict[name[idx]] = loss.cpu().item()
    print("valid dataset finish!!!")

train_sorted_loss = sorted(train_loss_dict.items(), key=lambda x: x[1])
print(len(train_sorted_loss))

train_loss_file = './results/new_cal_loss/googlenet_train_loss_file.txt'
f1 = open(train_loss_file, mode='w+', encoding='utf-8')
for name, loss in train_sorted_loss:
    f1.write(str(name))
    f1.write(':')
    f1.write(str(loss))
    f1.write("\n")
f1.close()

print("train dataset loss finish")

valid_sorted_loss = sorted(valid_loss_dict.items(), key=lambda x: x[1])
print(len(valid_sorted_loss))

valid_loss_file = './results/new_cal_loss/googlenet_valid_loss_file.txt'
f2 = open(valid_loss_file, mode='w+', encoding='utf-8')
for name, loss in valid_sorted_loss:
    f2.write(str(name))
    f2.write(':')
    f2.write(str(loss))
    f2.write("\n")
f2.close()

print("valid dataset loss finish")
