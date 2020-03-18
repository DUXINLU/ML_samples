import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

'''
x1 = Variable(torch.tensor([[1], [2]], dtype=torch.float), requires_grad=True)
x2 = Variable(torch.tensor([3, 4], dtype=torch.float), requires_grad=True)
print(x1, x2)

y = x1 * x2
print(y)
# y.backward(torch.ones(y.size()))
y.backward(torch.ones(y.size()) / 2)
print(x1.grad)
print(x2.grad)

# 构建一个y = x^2 函数 求x = 2 的导数
import numpy as np
import torch
from torch.autograd import Variable
# 1、画出函数图像
import matplotlib.pyplot as plt

x = np.arange(-3, 3.01, 0.1)
y = x ** 2
plt.plot(x, y)
plt.plot(2, 4, 'ro')
plt.show()

# 定义点variable类型的x = 2

x = Variable(torch.FloatTensor([2]), requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)'''

x = torch.tensor([i for i in range(10)], dtype=torch.float, requires_grad=True)
print(x)
print(F.softmax(x, dim=0))
