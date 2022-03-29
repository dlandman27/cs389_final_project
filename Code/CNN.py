import torch.nn
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = torch.nn.MaxPool2d((2,2), stride=(2,2))
        self.linear = nn.linear()
        self.relu = nn.ReLU()


    def forward(self, x):
        out = x

        return out