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

class dataset():
    def __init__(self,filepath,crop_width,crop_height,batch,train=True):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", filepath))
        self.batch = batch
        self.dataset, self.test_set, self.dataset_features, self.test_features = self.load_dataset(crop_width, crop_height, batch, train)
        self.train = train
    
    # Get Methods
    def getCropWidth(self):
        return self.crop_width
    
    def getCropHeight(self):
        return self.crop_height
    
    def getFilepath(self):
        return self.filepath
    
    def getBatch(self):
        return self.batch
    
    def getDataset(self):
        return self.dataset
    
    def getTestSet(self):
        return self.test_set
    
    def getDatasetFeatures(self):
        return self.dataset_features
    
    def getTestFeatures(self):
        return self.test_features
    
    def getTrain(self):
        return self.train

    # Loads the Dataset
    def load_dataset(self,crop_width, crop_height, batch_size, train=True):
        training_set_final = [] # This is the final training set
        test_set_final = [] # This is the final test set
        training_feature_set = [] # This is the final training feature set
        test_feature_set = [] # This is the final test feature set
        dataset = [] # This is an array of all the image URLs

        directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "thecarconnectionpicturedataset"))
        for file in tqdm(os.listdir(directory)):
            dataset.append(file)

        # Splits the dataset into training and test sets
        training_Set, test_set = train_test_split(dataset, test_size=0.3, random_state=25)

        # Crops the image and adds it to the feature set
        def process_Image_Set(data, final_set, final_feature):
            counter = 0
            batch_counter = 0
            batch = []
            feature_batch = []
            for image in data:
                img = Image.open(os.path.join(directory, image))

                # Crops the image to size new_width x new_height
                width = img.size[0]
                height = img.size[1]
                left = (width - crop_width) / 2
                top = (height - crop_height) / 2
                right = left + crop_width
                bottom = top + crop_height

                # Cropped image of above dimension
                # (It will not change original image)
                img = img.crop((left, top, right, bottom))
                img = np.asarray(img).reshape(3, 300, 210)

                img_features = np.array(image.split("_"))  # Gets the features from the file name
                if img_features[3] == "nan":
                    continue
                price = [int(img_features[3])] # Gets the price from the file name

                # TODO DELETE THIS
                if counter > 1000:
                    break
                else:
                    counter += 1
                if batch_counter < batch_size:
                    batch.append(img)
                    feature_batch.append(price)
                    batch_counter = batch_counter + 1
                else:
                    final_set.append(np.array(batch))
                    final_feature.append(np.array(feature_batch))
                    batch = []
                    feature_batch = []
                    batch_counter = 0

        process_Image_Set(training_Set, training_set_final, training_feature_set)
        process_Image_Set(test_set, test_set_final, test_feature_set)

        return (np.array(training_set_final), np.array(test_set_final), np.array(training_feature_set), np.array(test_feature_set))

    # Finds the average image dimension of the dataset
    def Find_Average_Image_Dimension(self):
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


    # Returns a random image from the dataset
    def getRandomImage(self):
        return self.dataset[random.randint(0, 10)]
    
    # Plots an image from the dataset given an image
    def plot_image(self, image):
        image = image.reshape(-1,self.crop_height,self.crop_width,3)
        plt.imshow(image[0])
        plt.show()

    # Plots a random image from the dataset
    def plotRandomImage(self):
        image = self.getRandomImage()
        self.plot_image(image)
