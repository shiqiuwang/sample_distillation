# -*- coding: utf-8 -*-

'''
@Time    : 2021/7/16 10:16
@Author  : Qiushi Wang
@FileName: sorted_10_100_hard_to_easy.py
@Software: PyCharm
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from with_index_my_dataset import RMBDataset
import os
import random

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
    model = models.alexnet(num_classes=257)
    pretrained_state_dict = torch.load(path_state_dict, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_state_dict.state_dict())

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    # model.to(device)
    return model


# 参数设置
BATCH_SIZE = 128

# ============================ step 1/5 数据 ============================

# split_dir = os.path.join("..", "..", "data", "rmb_split")
split_dir = "./data/caltech_split_data"
train_dir = os.path.join(split_dir, "train")
# valid_dir = os.path.join(split_dir, "valid")

path_state_dict = './pkl/finish_model.pkl'

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

# 构建DataLoder
train_loader = DataLoader(dataset=train_data_tensor, batch_size=BATCH_SIZE, shuffle=True)
print("len(train_data)", len(train_data))
print("len(train_loader)", len(train_loader))

# ============================ step 2/5 模型 ============================

alexnet_model = get_model(path_state_dict, False)
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
sorted_loss = sorted(loss_dict.items(), key=lambda x: x[1], reverse=True)
# sorted_sim = sorted(sim_dict.items(), key=lambda x: x[0])
# sorted_labels = dict(sorted(label_dict.items(), key=lambda x: x[0]))
print(len(sorted_loss))

easy_10_100 = sorted_loss[:int(len(sorted_loss) * 0.9)]

# random.shuffle(easy_10_100)
# test = sorted_loss[int(len(sorted_loss)*0.9):]
# medium = sorted_loss[int(len(sorted_loss)*0.3):int(len(sorted_loss)*0.6)]
# hard = sorted_loss[int(len(sorted_loss)*0.6):int(len(sorted_loss)*0.9)]
#
# test = sorted_loss[int(len(sorted_loss)*0.9):]
label_file = './results/different_learning_method/hard_to_easy/labels.txt'
f = open(label_file, mode='w+', encoding='utf-8')
for index, item in enumerate(easy_10_100):
    idx = item[0]
    choose_image = train_data[int(train_idx_to_real_index[idx])][1]
    choose_label = train_data[int(train_idx_to_real_index[idx])][2]
    save_path = './results/different_learning_method/hard_to_easy/images/' + str(int(index)) + '.jpg'
    choose_image.save(save_path)
    f.write(str(choose_label))
    f.write('\n')
f.close()

print('finish!!!')
