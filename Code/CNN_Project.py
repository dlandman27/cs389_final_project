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
from PIL import Image #not sure if this one is necessary

# Loads the dataset from the file location
def load_dataset(batch_size = 32, train=True):
    dataset = []
    batch_counter = 0
    batch = []
    feature_set = []
    counter = 0
    # TODO Add file size changes
    #/Users/maanasperi/Documents/VSCode/CS389/Final_Proj
    for file in tqdm(os.listdir('/Users/maanasperi/Documents/VSCode/CS389/Final_Proj/thecarconnectionpicturedataset')):
        img = Image.open('/Users/maanasperi/Documents/VSCode/CS389/Final_Proj/thecarconnectionpicturedataset/'+file).resize((89,109)) # Opens the Image
        img = np.asarray(img).reshape(3,109,89)
        img_features = file.split("_") # Gets the features from the file name
        feature_set.append(img_features)

        if(counter > 1000):
            break
        else:
            counter+=1
        if batch_counter < batch_size:
            batch.append(img)
            batch_counter = batch_counter + 1
        else:
            dataset.append(np.array(batch))
            batch = []
            batch_counter = 0

    return np.array(dataset)

def plot_image(image):
    image = image.reshape(-1,109,89,3)
    plt.imshow(image[0])
    plt.show()
    return


dataset = load_dataset(batch_size=32,train=True)
ex_image = dataset[0]
print("image shape:", ex_image.shape)
plot_image(ex_image)