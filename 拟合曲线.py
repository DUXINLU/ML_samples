import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def get_data(x, w, b, d):
    c, r = x.shape
    y = (w * x * x + b * x + d) + (0.1 * (2 * np.random.rand(c, r) - 1))
    return y


xs = np.arange(0, 3, 0.01).reshape(-1, 1)
ys = get_data(xs, 1, -2, 3)

xs = Variable(torch.Tensor(xs), requires_grad=True)
ys = Variable(torch.Tensor(ys), requires_grad=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)

        self.criterion = nn.MSELoss()
        self.optim = optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        return output


net = Net()

for epoch in range(5000):
    y_predict = net(xs)

    loss = net.criterion(y_predict, ys)
    if epoch % 100 == 0:
        print(epoch, loss)

    net.optim.zero_grad()
    loss.backward()
    net.optim.step()

ys_pre = net(xs)

plt.title("curve")
plt.plot(xs.data.numpy(), ys.data.numpy())
plt.plot(xs.data.numpy(), ys_pre.data.numpy())
plt.legend("ys", "ys_pre")
plt.show()
