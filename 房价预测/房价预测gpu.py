import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

# 读取、清洗数据
train = pd.read_csv('./house_price/train.csv')
test = pd.read_csv('./house_price/test.csv')
all_features = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
all_labels = pd.concat((train.loc[:, 'SalePrice'], test.loc[:, 'SalePrice']))
# 数值特征
numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
# 非数值特征
object_feats = all_features.dtypes[all_features.dtypes == "object"].index

all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = pd.get_dummies(all_features, prefix=object_feats, dummy_na=True)
# 空值：每一个特征的全局平均值来代替无效值 NA就是指空值
all_features = all_features.fillna(all_features.mean())

mean = all_labels.mean()
std = all_labels.std()
all_labels = (all_labels - mean) / std

num_train = train.shape[0]
train_features = all_features[:num_train].values.astype(np.float32)  # (1314, 331)
test_features = all_features[num_train:].values.astype(np.float32)  # (146, 331)
train_labels = all_labels[:num_train].values.astype(np.float32)
test_labels = all_labels[num_train:].values.astype(np.float32)

train_features = torch.from_numpy(train_features)
train_labels = torch.from_numpy(train_labels).unsqueeze(1)
test_features = torch.from_numpy(test_features)
test_labels = torch.from_numpy(test_labels).unsqueeze(1)
train_set = TensorDataset(train_features, train_labels)
test_set = TensorDataset(test_features, test_labels)
train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_data = DataLoader(dataset=test_set, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(331, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

        self.optim = optim.SGD(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        return y


# 将模型放入GPU
net = Net().cuda()

for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_data):
        # 将data,target放入GPU
        data_gpu, target_gpu = Variable(data.cuda()), Variable(target.cuda())

        net.optim.zero_grad()
        target_predict = net(data_gpu)
        loss = net.criterion(target_predict, target_gpu)
        loss.backward()
        net.optim.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_gpu), len(train_data.dataset),
                       100. * batch_idx / len(train_data), loss.item()))

total_loss = 0
for idx, (data, target) in enumerate(test_data):
    # data target放入GPU
    data_gpu, target_gpu = data.cuda(), target.cuda()
    target_predict = net(data_gpu)

    loss = net.criterion(target_predict, target_gpu)
    total_loss += loss.item()
print(total_loss / len(test_data))
