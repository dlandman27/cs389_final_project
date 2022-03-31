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

def Find_Average_Image_Dimension():
    width = 0
    height = 0
    number_Of_Images = 0
    directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "thecarconnectionpicturedataset"))
    for file in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory,file)
        img = Image.open(filepath) # Opens the Image
        number_Of_Images +=1
        width += img.size[0]
        height += img.size[1]
    width = width/number_Of_Images
    height = height/ number_Of_Images
    return (width,height)


# Loads the dataset from the file location
def load_dataset( crop_width, crop_height, batch_size = 32, train=True):
    dataset = []
    batch_counter = 0
    batch = []
    feature_set = []
    counter = 0
    # TODO Add file size changes
    directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "thecarconnectionpicturedataset"))
    for file in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory,file)
        img = Image.open(filepath) # Opens the Image

        # Crops the image to size new_width x new_height
        width = img.size[0]
        height = img.size[1]
        left = (width - crop_width)/2
        top = (height - crop_height)/2
        right = left+crop_width
        bottom = top+crop_height

        # Cropped image of above dimension
        # (It will not change original image)
        img = img.crop((left, top, right, bottom))
        img = np.asarray(img)
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




def plot_image(image, crop_width, crop_height):
    image = image.reshape(-1,crop_height,crop_width,3)
    plt.imshow(image[0])
    plt.show()
    return

#Since I found the average image size there is no need to run this function anymore
#print(Find_Average_Image_Dimension())


crop_width = 300  # Average Width of the cropped image is 320
crop_height = 210  # Average Height of the cropped image is 230
dataset = load_dataset(crop_width, crop_height, batch_size=32, train=True)

ex_image = dataset[random.randint(0, 10)]
print("image shape:", ex_image.shape)
plot_image(ex_image,crop_width=crop_width,crop_height=crop_height)