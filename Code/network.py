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


    def test_model(self, test_data, test_label):
        test_data = self.data.getDatasetFeatures()
        test_label = self.data.getDatasetLabels()
        sum_loss = 0
        n_correct = 0
        total = 0
        counter = 0

        for i in tqdm(test_data):
            # This is exactly the same as the training loop
            # without the, well, training, part
            tensor_label = torch.from_numpy(test_label[counter]).float()
            pred = self.model(i)
            loss = self.loss_function(pred, tensor_label)
            sum_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            n_correct += (predicted == tensor_label).sum()
            total += tensor_label.size(0)
            counter += 1

        test_acc = round(((n_correct / total).item() * 100), 2)
        avg_loss = round(sum_loss / len(test_data), 2)

        print("test accuracy:", test_acc)
        print("test loss:", avg_loss)

        return test_acc, avg_loss


    def train_and_test(self):
        trained_model, losses = self.train_model()

        test_acc, test_loss = self.test_model(trained_model, test_set, test_features)

        plt.plot(np.arange(len(losses)) * self.batch * self.update_interval, losses, label="training loss")
        plt.hlines(test_loss, 0, len(losses) * self.batch * self.update_interval, color='r', label="test loss")
        plt.title("training curve")
        plt.xlabel("number of images trained on")
        plt.ylabel("Reconstruction loss")
        path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "graphs", "training2.png"))
        plt.savefig(path)
        plt.show()

        return trained_model, test_loss