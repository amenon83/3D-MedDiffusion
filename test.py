test = "Hello, World!"
print(test)


import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()
print(net)          