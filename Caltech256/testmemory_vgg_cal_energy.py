# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/12 21:05
@Author  : Qiushi Wang
@FileName: vgg_cal_energy.py
@Software: PyCharm
'''

# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/5 23:11
@Author  : Qiushi Wang
@FileName: vgg_train_different_set_for_the_same_testset.py
@Software: PyCharm
'''

# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/5 16:33
@Author  : Qiushi Wang
@FileName: resnet_train_different_set_for_the_same_testset.py
@Software: PyCharm
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
# from matplotlib import pyplot as plt
# from model.lenet import LeNet
import torchvision.models as models
from my_dataset import RMBDataset
from my_dataset_for_hard_sample import RMBHardDataset
import time
# from SPLD import spld
# from mcmc import MC
from early_stopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import energyusage
#import torchstat
from torchstat import stat
import torchvision.models as models

warnings.filterwarnings("ignore")

device = torch.device("cuda:0")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容




def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    stat(model, (3, 256, 256))
    model.to(device)
    
    return model


set_seed()  # 设置随机种子
# rmb_label = {}
# for i in range(100):
#    rmb_label[str(i)] = i

# 参数设置
MAX_EPOCH = 1
BATCH_SIZE = 64
LR = 0.001
log_interval = 1
val_interval = 1

# T = 10000
#
# # 自步学习参数
# lam = 0.1
# gamma = 0.25
#
# u1 = 1.1
# u2 = 1.1
num_classes = 257
# samples_label_arr = np.array(list(range(0, 100)))

# ============================ step 1/5 数据 ============================

# split_dir = os.path.join("..", "..", "data", "rmb_split")
# split_dir = "./data/imagenet_split_data"
train_dir = ['./results/confusion_dataset/vgg/80_90',
             './results/confusion_dataset/vgg/90_100']
valid_dir = "./data/caltech_split_data/valid"
# train_dir = os.path.join(split_dir, "train2")
# valid_dir = os.path.join(split_dir, "valid")

path_state_dict = './data/vgg16-397923af.pth'
# num_classes = 100

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256)),  # (256, 256) 区别
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

normalizes = transforms.Normalize(norm_mean, norm_std)

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.TenCrop(224, vertical_flip=False),
    transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
])

# 构建MyDataset实例
train_data = RMBHardDataset(data_dirs=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print(len(train_data))
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
print(len(valid_data))
# device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
# ============================ step 2/5 模型 ============================
# 初始化v_star
# v_star = torch.randint(low=0, high=2, size=(len(train_loader), BATCH_SIZE), dtype=torch.float)
alexnet_model = get_model(path_state_dict, False)
#torchstat.stat(alexnet_model, (3, 256, 256))
#print(alexnet_model)
num_ftrs = alexnet_model.classifier._modules["6"].in_features
alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)

alexnet_model = alexnet_model.to(device)
# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()
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
optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
# train_curve = list()
# valid_curve = list()
# select_list = []

valid_acc = []
valid_precision = []
valid_recall = []
valid_f1 = []
time_record = []

train_acc = []
train_precision = []
train_recall = []
train_f1 = []
train_time_record = []


def train(MAX_EPOCH):
    start_time = time.time()
    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        y_true = []
        y_pred = []
        y_valid_true = []
        y_valid_pred = []
        alexnet_model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = alexnet_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            for la in labels.cpu().numpy():
                y_true.append(la)
            for pre in predicted.cpu().numpy():
                y_pred.append(pre)
            # 打印训练信息
            loss_mean += loss.item()
            # train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                train_acc.append(correct / total)
                train_precision.append(precision_score(y_true, y_pred, average='macro'))
                train_recall.append(recall_score(y_true, y_pred, average='macro'))
                train_f1.append(f1_score(y_true, y_pred, average='macro'))
                train_time_record.append(time.time() - start_time)
                print(
                    "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                        epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total,
                        precision_score(y_true, y_pred, average='macro'),
                        recall_score(y_true, y_pred, average='macro'),
                        f1_score(y_true, y_pred, average='macro')))
                loss_mean = 0.
        scheduler.step()

        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            alexnet_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    bs, ncrops, c, h, w = inputs.size()  # [4, 10, 3, 224, 224
                    outputs = alexnet_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    for la in labels.cpu().numpy():
                        y_valid_true.append(la)
                    for pre in predicted.cpu().numpy():
                        y_valid_pred.append(pre)

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)

                valid_acc.append(correct_val / total_val)
                valid_precision.append(precision_score(y_valid_true, y_valid_pred, average='macro'))
                valid_recall.append(recall_score(y_valid_true, y_valid_pred, average='macro'))
                valid_f1.append(f1_score(y_valid_true, y_valid_pred, average='macro'))
                time_record.append(time.time() - start_time)
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                        epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val,
                        precision_score(y_valid_true, y_valid_pred, average='macro'),
                        recall_score(y_valid_true, y_valid_pred, average='macro'),
                        f1_score(y_valid_true, y_valid_pred, average='macro')))
                # early_stopping(loss_val_mean, alexnet_model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     # 结束模型训练
                #     break
    end_time = time.time()
    print("running time is:", end_time - start_time)
    #return end_time - start_time


if __name__ == '__main__':
     train(MAX_EPOCH)
    #energyusage.evaluate(train,1, pdf=True, energyOutput=False)

# 将结果写入文本
# with open(file='./results/hard_sample_train/vgg/10_100/valid_acc.txt', mode='w+',
#           encoding='utf-8') as f1:
#     for val in valid_acc:
#         f1.write(str(val))
#         f1.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/valid_precision.txt', mode='w+',
#           encoding='utf-8') as f2:
#     for val in valid_precision:
#         f2.write(str(val))
#         f2.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/valid_recall.txt', mode='w+',
#           encoding='utf-8') as f3:
#     for val in valid_recall:
#         f3.write(str(val))
#         f3.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/valid_f1.txt', mode='w+',
#           encoding='utf-8') as f4:
#     for val in valid_f1:
#         f4.write(str(val))
#         f4.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/valid_time_record.txt', mode='w+',
#           encoding='utf-8') as f5:
#     for val in time_record:
#         f5.write(str(val))
#         f5.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/train_acc.txt', mode='w+',
#           encoding='utf-8') as f6:
#     for val in train_acc:
#         f6.write(str(val))
#         f6.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/train_precision.txt', mode='w+',
#           encoding='utf-8') as f7:
#     for val in train_precision:
#         f7.write(str(val))
#         f7.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/train_recall.txt', mode='w+',
#           encoding='utf-8') as f8:
#     for val in train_recall:
#         f8.write(str(val))
#         f8.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/train_f1.txt', mode='w+',
#           encoding='utf-8') as f9:
#     for val in train_f1:
#         f9.write(str(val))
#         f9.write('\n')
#
# with open(file='./results/hard_sample_train/vgg/10_100/train_time_record.txt', mode='w+',
#           encoding='utf-8') as f10:
#     for val in train_time_record:
#         f10.write(str(val))
#         f10.write('\n')
