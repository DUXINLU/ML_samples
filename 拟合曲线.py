import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def get_data(x, a, b, c, d):
    h, w = x.shape
    y = (a * x * x * x + b * x * x + c * x + d) + (0.1 * (2 * np.random.rand(h, w) - 1))
    return y


xs = np.arange(0, 5, 0.01).reshape(-1, 1)
ys = get_data(xs, 0, -2, 3, -2)

xs = Variable(torch.Tensor(xs), requires_grad=True)
ys = Variable(torch.Tensor(ys), requires_grad=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)

        self.criterion = nn.MSELoss()
        self.optim = optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
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
