import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, bias=False)  # 26x26x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, bias=False)  # 24x24x16
        self.pool = nn.MaxPool2d(2, 2)  # 12x12x16
        #self.fc1 = nn.Linear(12 * 12 * 16, 128)
        #self.fc2 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(12 * 12 * 16, 10, bias=False)
        self.fc2 = nn.Linear(10, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 12 * 12 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x