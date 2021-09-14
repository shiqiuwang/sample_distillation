# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/12 0:25
@Author  : Qiushi Wang
@FileName: resnet_split_to_ten_parts.py
@Software: PyCharm
'''

# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/11 9:38
@Author  : Qiushi Wang
@FileName: vgg_split_to_ten_parts.py
@Software: PyCharm
'''

# -*- coding: utf-8 -*-
"""
# @file name  : train_lenet.py
# @author     : tingsongyu
# @date       : 2019-09-07 10:08:00
# @brief      : 人民币分类模型训练
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
# from model.lenet import LeNet
import torchvision.models as models
from with_index_my_dataset import RMBDataset
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4'
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
    model = models.resnet50(num_classes=257)
    pretrained_state_dict = torch.load(path_state_dict, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_state_dict.state_dict())

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    # model.to(device)
    return model


# set_seed()  # 设置随机种子
# rmb_label = {}
# fo r i in range(100):
#     rmb_label[str(i)] = i

# 参数设置
BATCH_SIZE = 128

# ============================ step 1/5 数据 ============================

# split_dir = os.path.join("..", "..", "data", "rmb_split")
split_dir = "./data/caltech_split_data"
train_dir = os.path.join(split_dir, "train")
# valid_dir = os.path.join(split_dir, "valid")

path_state_dict = './pkl/finish_model_resnet.pkl'

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

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir)
train_data_tensor = RMBDataset.image2tensor(train_data, train_transform)
# valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data_tensor, batch_size=BATCH_SIZE, shuffle=True)
print("len(train_data)", len(train_data))
print("len(train_loader)", len(train_loader))
# valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
# ============================ step 2/5 模型 ============================
# 初始化v_star
# v_star = torch.randint(low=0, high=2, size=(len(train_loader), BATCH_SIZE), dtype=torch.float)
alexnet_model = get_model(path_state_dict, False)
# alexnet_model = nn.DataParallel(alexnet_model)
alexnet_model = alexnet_model.to(device)

# num_ftrs = alexnet_model.classifier._modules["6"].in_features
# alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)
#
# alexnet_model.to(device)
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
# optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
# train_curve = list()
# valid_curve = list()
# select_list = []

data_idx_dict = dict()
select_idx_list = set()
train_idx_to_real_index = {}
loss_dict = dict()  # index:loss
sim_dict = dict()  # index:similarity
label_dict = dict()  # index:label

alexnet_model.eval()
count = 0
with torch.no_grad():
    for i, data in enumerate(train_loader):
        index, inputs, labels = data
        labels = labels.view(-1, 1)
        # if hasattr(torch.cuda, "empty_cache"):
        #     torch.cuda.empty_cache()
        inputs = inputs.to(device)
        count += 1
        print(count)
        labels = labels.to(device)

        for idx in range(len(inputs)):
            # sim_dict[i * BATCH_SIZE + idx] = inputs[idx].cpu().numpy().var()
            # data_idx_dict[i * BATCH_SIZE + idx] = [inputs[idx], labels[idx]]
            train_idx_to_real_index[i * BATCH_SIZE + idx] = index[idx].cpu().item()

            outputs = alexnet_model(inputs[idx].unsqueeze_(0))
            loss = criterion(outputs, labels[idx])
            loss_dict[i * BATCH_SIZE + idx] = loss.cpu().item()
            # label_dict[i * BATCH_SIZE + idx] = labels[idx].cpu().item()
sorted_loss = sorted(loss_dict.items(), key=lambda x: x[1])
# sorted_sim = sorted(sim_dict.items(), key=lambda x: x[0])
# sorted_labels = dict(sorted(label_dict.items(), key=lambda x: x[0]))
print(len(sorted_loss))

# easy = sorted_loss[:int(len(sorted_loss)*0.3)]
easy_0_10 = sorted_loss[:int(len(sorted_loss) * 0.1)]
easy_10_20 = sorted_loss[int(len(sorted_loss) * 0.1):int(len(sorted_loss) * 0.2)]
easy_20_30 = sorted_loss[int(len(sorted_loss) * 0.2):int(len(sorted_loss) * 0.3)]
easy_30_40 = sorted_loss[int(len(sorted_loss) * 0.3):int(len(sorted_loss) * 0.4)]
easy_40_50 = sorted_loss[int(len(sorted_loss) * 0.4):int(len(sorted_loss) * 0.5)]
easy_50_60 = sorted_loss[int(len(sorted_loss) * 0.5):int(len(sorted_loss) * 0.6)]
easy_60_70 = sorted_loss[int(len(sorted_loss) * 0.6):int(len(sorted_loss) * 0.7)]
easy_70_80 = sorted_loss[int(len(sorted_loss) * 0.7):int(len(sorted_loss) * 0.8)]
easy_80_90 = sorted_loss[int(len(sorted_loss) * 0.8):int(len(sorted_loss) * 0.9)]
easy_90_100 = sorted_loss[int(len(sorted_loss) * 0.9):]
# test = sorted_loss[int(len(sorted_loss)*0.9):]
# medium = sorted_loss[int(len(sorted_loss)*0.3):int(len(sorted_loss)*0.6)]
# hard = sorted_loss[int(len(sorted_loss)*0.6):int(len(sorted_loss)*0.9)]
#
# test = sorted_loss[int(len(sorted_loss)*0.9):]

for item in easy_0_10:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/0_10/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

print("finish 0_10!!!")

for item in easy_10_20:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/10_20/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
print("finish 10_20!!!")

for item in easy_20_30:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/20_30/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

print("finish 20_30!!!")

for item in easy_30_40:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/30_40/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
print("finish 30_40!!!")

for item in easy_40_50:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/40_50/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
print("finish 40_50!!!")

for item in easy_50_60:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/50_60/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

print("finish 50_60!!!")

for item in easy_60_70:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/60_70/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
print("finish 60_70!!!")

for item in easy_70_80:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/70_80/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

print("finish 70_80!!!")

for item in easy_80_90:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/80_90/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
print("finish 80_90!!!")

for item in easy_90_100:
    idx = item[0]
    if train_data[int(train_idx_to_real_index[idx])][2] == 0:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/0/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 1:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/1/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 2:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/2/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 3:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/3/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 4:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/4/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 5:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/5/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 6:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/6/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 7:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/7/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 8:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/8/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 9:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/9/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 10:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/10/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 11:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/11/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 12:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/12/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 13:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/13/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 14:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/14/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 15:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/15/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 16:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/16/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 17:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/17/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 18:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/18/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 19:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/19/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 20:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/20/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 21:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/21/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 22:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/22/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 23:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/23/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 24:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/24/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 25:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/25/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 26:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/26/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 27:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/27/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 28:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/28/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 29:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/29/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 30:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/30/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 31:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/31/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 32:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/32/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 33:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/33/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 34:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/34/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 35:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/35/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 36:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/36/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 37:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/37/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 38:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/38/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 39:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/39/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 40:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/40/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 41:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/41/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 42:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/42/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 43:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/43/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 44:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/44/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 45:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/45/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 46:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/46/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 47:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/47/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 48:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/48/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 49:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/49/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 50:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/50/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 51:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/51/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 52:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/52/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 53:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/53/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 54:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/54/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 55:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/55/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 56:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/56/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 57:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/57/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 58:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/58/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 59:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/59/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 60:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/60/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 61:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/61/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 62:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/62/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 63:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/63/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 64:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/64/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 65:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/65/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 66:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/66/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 67:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/67/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 68:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/68/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 69:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/69/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 70:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/70/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 71:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/71/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 72:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/72/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 73:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/73/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 74:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/74/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 75:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/75/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 76:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/76/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 77:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/77/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 78:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/78/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 79:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/79/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 80:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/80/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 81:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/81/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 82:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/82/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 83:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/83/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 84:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/84/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 85:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/85/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 86:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/86/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 87:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/87/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 88:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/88/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 89:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/89/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 90:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/90/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 91:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/91/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 92:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/92/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 93:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/93/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 94:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/94/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 95:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/95/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 96:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/96/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 97:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/97/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 98:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/98/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 99:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/99/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 100:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/100/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 101:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/101/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 102:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/102/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 103:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/103/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 104:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/104/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 105:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/105/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 106:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/106/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 107:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/107/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 108:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/108/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 109:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/109/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 110:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/110/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 111:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/111/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 112:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/112/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 113:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/113/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 114:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/114/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 115:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/115/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 116:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/116/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 117:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/117/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 118:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/118/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 119:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/119/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 120:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/120/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 121:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/121/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 122:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/122/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 123:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/123/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 124:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/124/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 125:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/125/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 126:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/126/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 127:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/127/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 128:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/128/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 129:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/129/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 130:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/130/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 131:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/131/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 132:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/132/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 133:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/133/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 134:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/134/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 135:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/135/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 136:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/136/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 137:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/137/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 138:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/138/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 139:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/139/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 140:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/140/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 141:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/141/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 142:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/142/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 143:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/143/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 144:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/144/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 145:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/145/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 146:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/146/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 147:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/147/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 148:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/148/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 149:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/149/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 150:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/150/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 151:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/151/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 152:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/152/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 153:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/153/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 154:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/154/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 155:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/155/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 156:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/156/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 157:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/157/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 158:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/158/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 159:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/159/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 160:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/160/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 161:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/161/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 162:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/162/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 163:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/163/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 164:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/164/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 165:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/165/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 166:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/166/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 167:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/167/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 168:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/168/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 169:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/169/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 170:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/170/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 171:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/171/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 172:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/172/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 173:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/173/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 174:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/174/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 175:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/175/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 176:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/176/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 177:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/177/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 178:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/178/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 179:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/179/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 180:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/180/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 181:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/181/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 182:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/182/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 183:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/183/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 184:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/184/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 185:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/185/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 186:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/186/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 187:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/187/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 188:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/188/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 189:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/189/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 190:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/190/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 191:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/191/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 192:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/192/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 193:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/193/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 194:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/194/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 195:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/195/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 196:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/196/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 197:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/197/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 198:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/198/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 199:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/199/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 200:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/200/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 201:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/201/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 202:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/202/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 203:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/203/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 204:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/204/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 205:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/205/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 206:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/206/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 207:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/207/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 208:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/208/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 209:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/209/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 210:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/210/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 211:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/211/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 212:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/212/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

    if train_data[int(train_idx_to_real_index[idx])][2] == 213:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/213/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 214:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/214/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 215:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/215/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 216:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/216/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 217:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/217/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 218:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/218/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 219:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/219/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 220:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/220/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 221:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/221/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 222:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/222/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 223:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/223/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 224:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/224/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 225:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/225/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 226:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/226/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 227:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/227/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 228:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/228/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 229:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/229/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 230:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/230/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 231:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/231/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 232:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/232/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 233:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/233/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 234:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/234/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 235:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/235/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 236:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/236/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 237:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/237/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 238:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/238/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 239:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/239/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 240:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/240/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 241:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/241/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 242:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/242/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 243:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/243/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 244:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/244/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 245:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/245/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 246:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/246/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 247:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/247/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 248:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/248/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 249:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/249/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 250:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/250/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 251:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/251/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 252:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/252/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 253:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/253/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 254:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/254/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 255:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/255/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)
    if train_data[int(train_idx_to_real_index[idx])][2] == 256:
        choose_image = train_data[int(train_idx_to_real_index[idx])][1]
        save_path = './results/confusion_dataset/resnet/90_100/256/' + str(int(train_idx_to_real_index[idx])) + '.jpg'
        choose_image.save(save_path)

print("finish 90_100!!!")
