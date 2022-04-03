import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import make_grid
from PIL import Image


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #the inital image shape is (3, 300, 210)

        #Stride: 1
        #Kernel Size: 5x5 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 64, kernel_size=(7, 7))  # (64, (294,204)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))  # (64, (147,102))
        self.conv2 = nn.Conv2d(64, 128, 5)  # (128, (143,98))
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(1, 0))  # (128, (72,49))
        self.conv3 = nn.Conv2d(128, 256, (3, 3))  # (256, (70,47))
        self.pool3 = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 1))  # (256, (35,24))
        # self.drop = nn.dropout()
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256*35*24, 100)
        self.fc2 = nn.Linear(100, 1)





    def forward(self, x):

        x = torch.from_numpy(x).float()
        x = x.reshape(-1, 3, 300, 210)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.5)
        # x = F.relu(self.fc2(x))
        #
        # print(x.shape())

        return x


