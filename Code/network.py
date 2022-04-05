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

class network():
    def __init__(self,data,loadModel,saveModel,batch,n_epochs,learning_rate,update_interval,loss_function,model,optimizer):
        self.data = data
        self.loadModel = loadModel
        self.saveModel = saveModel
        self.batch = batch
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.loss_function = loss_function
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr= learning_rate)
        

    def load_model(self):
        self.model = self.model.load_state_dict(torch.load("../SavedStates/ModelState.pt"))

    def save_model(self,r):
        direct = os.path.realpath(os.path.join(os.path.dirname("ModelState.pt"), "..", "SavedStates"))
        torch.save(r[0].state_dict(), direct)

    def train_model(self):
        losses = []
        for n in range(self.n_epochs):
            iterator = 0
            count = 0
            for i in (tqdm(self.data.getDataset())):
                self.optimizer.zero_grad()
                
                my_output = self.model(i)

                # print(torch.from_numpy(featureSet[count]).float().shape)
                loss = self.loss_function(my_output, torch.from_numpy(self.data.getDatasetFeatures()[count]).float())
                loss.backward()
                self.optimizer.step()
                count = count + 1

                if iterator % self.update_interval == 0:
                    losses.append(round(loss.item(), 2))
                iterator+=1

        if(self.saveModel):
            self.save_model(losses)

        return losses