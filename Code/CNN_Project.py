from operator import xor
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import random
from tqdm import tqdm
from PIL import Image
from CNN import CNN
from sklearn.model_selection import train_test_split
from dataset import dataset
from network import network

# Hyperparamters
loadModel = False
saveModel = True
batch = 32
n_epochs = 1
learning_rate = .00001
update_interval = 100
loss_function = nn.CrossEntropyLoss()
crop_width = 300  # Average Width of the cropped image is 320
crop_height = 210  # Average Height of the cropped image is 230
optimizer = torch.optim.Adam
model = CNN()

data = dataset(filepath="thecarconnectionpicturedataset",crop_width=crop_width,crop_height=crop_height,batch=batch,train=True)
neural_network = network(data,loadModel,saveModel,batch,n_epochs,learning_rate,update_interval,loss_function,model,optimizer)

# data.plotRandomImage()

if(loadModel):
    neural_network.load_model()
else:
    neural_network.train_model()


#model.eval()







































#Option2: 
#Saving & Loading a General Checkpoint for Inference and/or Resuming Training
#https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training 
    #Saving
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#             }, "../SavedStates/ModelState.pt")

    #Loading
# checkpoint = torch.load("../SavedStates/ModelState.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.eval()
# # - or -
# model.train()
