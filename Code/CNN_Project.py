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
            if counter > 2000:
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
update_interval = 10
loss_function = nn.MSELoss()
crop_width = 300  # Average Width of the cropped image is 320
crop_height = 210  # Average Height of the cropped image is 230
dataset, test_set, dataset_features, test_features = load_dataset(crop_width, crop_height, batch_size=batch, train=True)
model = CNN()
directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "SavedStates", "ModelState.pt"))
model = model.load_state_dict(torch.load(directory)) if loadModel else model  # loads a model if you want
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ex_image = dataset[random.randint(0, 10)]  # prints an random image from the dataset and plots it
plot_image(ex_image, crop_width=crop_width, crop_height=crop_height)


def training(model, dataset, featureSet, loss_function, optimizer, n_epochs, update_interval):
    losses = []
    for n in range(n_epochs):
        iterator = 0
        count = 0
        for i in tqdm(dataset):
            optimizer.zero_grad()
            my_output = model(i)
            loss = loss_function(my_output, torch.from_numpy(featureSet[count]).float())
            loss.backward()
            optimizer.step()
            count = count + 1

            if iterator % update_interval == 0:
                losses.append(round(loss.item(), 2))
            iterator+=1

    return model, losses


def testing(model, loss_function, test_data, test_label):
    sum_loss = 0
    n_correct = 0
    total = 0
    counter = 0
    for i in tqdm(test_data):
        # This is essentially exactly the same as the training loop
        # without the, well, training, part
        tensor_label = torch.from_numpy(test_label[counter]).float()
        pred = model(i)
        loss = loss_function(pred, tensor_label)
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


def train_and_test(model, batch_size, dataset, dataset_features, test_set, test_features, loss_function, optimizer, n_epochs, update_interval):
    trained_model, losses = training(model, dataset, dataset_features, loss_function, optimizer, n_epochs, update_interval)

    test_acc, test_loss = testing(trained_model, loss_function, test_set, test_features)

    plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses, label="training loss")
    plt.hlines(test_loss, 0, len(losses) * batch_size * update_interval, color='r', label="test loss")
    plt.title("training curve")
    plt.xlabel("number of images trained on")
    plt.ylabel("Reconstruction loss")
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "graphs", "training2.png"))
    plt.savefig(path)
    plt.show()

    return trained_model, test_loss


if not loadModel:
    model, test_loss = train_and_test(model, batch, dataset, dataset_features, test_set, test_features, loss_function, optimizer, n_epochs, update_interval)

#save model
if saveModel:
    direct = os.path.realpath(os.path.join(os.path.dirname("ModelState.pt"), "..", "SavedStates", "ModelState.pt"))
    torch.save(model.state_dict(), direct)









































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
