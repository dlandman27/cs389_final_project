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


# Finds the average width and height of a givenn set of images
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

    training_set_final = [] # This is the final training set
    test_set_final = [] # This is the final test set
    training_feature_set = [] # This is the final training feature set
    test_feature_set = [] # This is the final test feature set
    dataset = [] # This is an array of all the image URLs

    directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "thecarconnectionpicturedataset"))
    for file in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, file)
        dataset.append(filepath)

    # Splits the dataset into training and test sets
    training_Set, test_set = train_test_split(dataset, test_size=0.3, random_state=25)

    # Crops the image and adds it to the feature set
    def process_Image_Set(data, final_set, final_feature):
        counter = 0
        batch_counter = 0
        batch = []
        feature_batch = []
        for image in dataset:
            img = Image.open(image)

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

            img_features = np.array(file.split("_"))  # Gets the features from the file name
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


def plot_image(image, crop_width, crop_height):
    image = image.reshape(-1,crop_height,crop_width,3)
    plt.imshow(image[0])
    plt.show()
    return

# Hyperparamters and Stuff
loadModel = False
saveModel = True
batch = 32
n_epochs = 1
learning_rate = .00001
update_interval = 1
loss_function = nn.CrossEntropyLoss()
crop_width = 300  # Average Width of the cropped image is 320
crop_height = 210  # Average Height of the cropped image is 230
dataset, test_set, dataset_features, test_features = load_dataset(crop_width, crop_height, batch_size=batch, train=True)
model = CNN()
directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "thecarconnectionpicturedataset"))
model = model.load_state_dict(torch.load(directory)) if loadModel else model  # loads a model if you want
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
ex_image = dataset[random.randint(0, 10)]  # prints an random image from the dataset and plots it
plot_image(ex_image, crop_width=crop_width, crop_height=crop_height)


def training(model, dataset, featureSet, loss_function, optimizer, n_epochs, update_interval):
    losses = []
    for n in range(n_epochs):
        iterator = 0
        count = 0
        for i in (tqdm(dataset)):
            optimizer.zero_grad()
            
            my_output = model(i)

            # print(torch.from_numpy(featureSet[count]).float().shape)
            loss = loss_function(my_output, torch.from_numpy(featureSet[count]).float())
            loss.backward()
            optimizer.step()
            count = count + 1

            if iterator % update_interval == 0:
                losses.append(round(loss.item(), 2))
            iterator+=1

    return model, losses



#Option1:
#Also variable methods when working with GPU, would look into that
if  not loadModel:
    model, losses = training(model, dataset, dataset_features, loss_function, optimizer, n_epochs, update_interval)
# direct = os.path.realpath(os.path.join(os.path.dirname("ModelState.pt"), "..", "SavedStates"))
#save model
if saveModel:
    direct = os.path.realpath(os.path.join(os.path.dirname("ModelState.pt"), "..", "SavedStates"))
    torch.save(model.state_dict(), direct)


plt.plot(np.arange(len(losses)) * batch * update_interval, losses)
plt.title("training curve")
plt.xlabel("number of images trained on")
plt.ylabel("Reconstruction loss")
plt.show()
directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "graphs"))






































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
