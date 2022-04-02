import torch.nn
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #the inital image shape is (3, 310, 210)

        #Stride: 1
        #Kernel Size: 5x5 
        
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        
        self.fc1 = nn.Linear(3*7*128, 1024) 
        self.fc2 = nn.Linear(1024, 2)





    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))

        print(x.shape())

        return x


