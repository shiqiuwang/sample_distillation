# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/19 8:44
@Author  : Qiushi Wang
@FileName: spld_train.py
@Software: PyCharm
'''

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from my_dataset import RMBDataset
import time
from SPLD import spld
from early_stopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from focalloss import FocalLoss

warnings.filterwarnings("ignore")

device = torch.device("cuda:2")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


patience = 5  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


set_seed()  # 设置随机种子
# rmb_label = {}
# for i in range(100):
#    rmb_label[str(i)] = i

# 参数设置
MAX_EPOCH = 20
BATCH_SIZE = 128
LR = 0.0001
log_interval = 1
val_interval = 1
num_classes = 257

# 自步学习参数
lam = 2
gamma = 2

beta = 0

u1 = 0.5
u2 = 0.5

# samples_label_arr = np.array(list(range(0, 100)))

# ============================ step 1/5 数据 ============================

# split_dir = os.path.join("..", "..", "data", "rmb_split")
# split_dir = "./data/imagenet_split_data"
train_dir = "./data/caltech_split_data/train"
valid_dir = "./data/caltech_split_data/valid"
# train_dir = os.path.join(split_dir, "train2")
# valid_dir = os.path.join(split_dir, "valid")

path_state_dict = './data/alexnet-owt-4df8aa71.pth'
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
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print("len(train_data):", len(train_data))
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
print("len(valid_data):", len(valid_data))

# 初始化v_star
v_star = torch.randint(low=0, high=2, size=(len(train_loader), BATCH_SIZE), dtype=torch.float)

# ============================ step 2/5 模型 ============================
alexnet_model = get_model(path_state_dict, False)

num_ftrs = alexnet_model.classifier._modules["6"].in_features
alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)

alexnet_model = alexnet_model.to(device)
# ============================ step 3/5 损失函数 ============================
criterion = FocalLoss(0)
# ============================ step 4/5 优化器 ============================
optimizer = optim.Adam(alexnet_model.parameters(), lr=LR, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
# train_curve = list()
# valid_curve = list()
# select_list = []

valid_acc = []
valid_precision = []
valid_recall = []
valid_f1 = []
time_record = []

select_idx_list = []
samples_label_arr = np.array(list(range(0, 256)))
record = []
types = list(range(0, 256))

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
        samples_label = labels.numpy().tolist()  # 拿到当前batch中样本的所有标签

        inputs = inputs.to(device)
        labels = labels.to(device)
        v_star[i] = v_star[i].to(device)

        outputs = alexnet_model(inputs)

        # backward
        optimizer.zero_grad()
        loss = (1 - beta) / (1 - beta ** sum(v_star[i])) * criterion(outputs, labels)

        loss1 = torch.matmul(v_star[i], loss.cpu()) - lam * v_star[i].sum()

        loss2 = torch.tensor(0, dtype=torch.float32)

        for group_id in samples_label_arr:
            idx_for_each_group = np.where(samples_label == group_id)[0]
            loss_for_each_group = torch.tensor(0, dtype=torch.float32)
            for idx in idx_for_each_group:
                loss_for_each_group += (v_star[i][idx] ** 2)
            loss2 += torch.sqrt(loss_for_each_group)
        loss2.to(device)

        # 计算E
        E = loss1 - gamma * loss2
        E.backward()
        # if clip > 0:
        #     torch.nn.utils.clip_grad_norm_(alexnet_model.parameters(), max_norm=20, norm_type=2)

        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        for la in labels.cpu().numpy():
            y_true.append(la)
        for pre in predicted.cpu().numpy():
            y_pred.append(pre)
        # 打印训练信息
        loss_mean += loss.mean().item()

        new_out_pre = alexnet_model(inputs)

        new_loss = criterion(new_out_pre, labels)
        # print("第{}个epoch的第{}个batch对应的loss是{}".format(epoch, i, new_loss.cpu().detach().numpy()))
        # new_loss = torch.mul(new_loss,-1)
        selected_idx_arr = spld(new_loss.reshape(new_loss.size()[0], ), samples_label, lam, gamma)
        # print("第{}个epoch的第{}个batch下选中的样本的索引是{}".format(epoch, i, selected_idx_arr))
        select_idx_list.append(selected_idx_arr)

        v_star[i] = torch.zeros((new_loss.size()[0],), dtype=torch.float32)

        for selected_idx in selected_idx_arr:
            v_star[i][selected_idx] = 1

        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print(
                "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total,
                    precision_score(y_true, y_pred, average='macro'),
                    recall_score(y_true, y_pred, average='macro'),
                    f1_score(y_true, y_pred, average='macro')))
            loss_mean = 0.
    scheduler.step()
    lam = u1 * lam
    gamma = u2 * gamma

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

                loss_val += loss.mean().item()

            loss_val_mean = loss_val / len(valid_loader)

            valid_acc.append(correct_val / total_val)
            valid_precision.append(precision_score(y_valid_true, y_valid_pred, average='macro'))
            valid_recall.append(recall_score(y_valid_true, y_valid_pred, average='macro'))
            valid_f1.append(f1_score(y_valid_true, y_valid_pred, average='macro'))
            # time_record.append(time.time() - start_time)
            print(
                "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val,
                    precision_score(y_valid_true, y_valid_pred, average='macro'),
                    recall_score(y_valid_true, y_valid_pred, average='macro'),
                    f1_score(y_valid_true, y_valid_pred, average='macro')))
            early_stopping(loss_val_mean, alexnet_model)
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break
end_time = time.time()
print("running time is:", end_time - start_time)
