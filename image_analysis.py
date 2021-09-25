# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 13:54:56 2021

@author: Arne
"""
import os
import time
import torch
import pandas as pd
import numpy as np
import pathlib
import PIL


from torch.utils.data import DataLoader



# data_loader = torch.utils.data.DataLoader(yesno_data,
#                                           batch_size=1,
#                                           shuffle=True)

import torch
from sklearn.preprocessing import LabelEncoder as LE

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image
class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = annotations_file
        self.img_dir = img_dir
        
        self.target_transform = target_transform
        self.len=self.__len__()
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = np.array(Image.open(img_path))
        label = self.img_labels.iloc[idx, 1]
        label=torch.tensor(label, dtype=torch.long, device=device)
        transform=transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor()], 
                                      )
                                      #transforms.Normalize(np.mean(image), np.std(image))])
        image =transform(image)
        
        return image/255, label
    
    
    def get_splits(self, n_test=50):
    # def get_splits(self, n_test=200):
        # determine sizes
        test_size = n_test  # validation
        train_size = self.len - test_size  # training

        # calculate the split indexes
        return random_split(self, [train_size, test_size])
    
    


import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
CPUdevice = torch.device("cpu")

import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
# from torchvision.models.segmentation import deeplabv3_resnet50 as resnet
# model=
import time
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from tqdm import tqdm
import shutil

import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# train_loader=DataLoader(traindata, batch_size=16, shuffle=True)
###next step: set up model and start the fit


# device casting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


# load a checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['mse']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
class ownCNN(nn.Module):
    #good explanation of what the layers do:
        #https://www.pluralsight.com/guides/image-classification-with-pytorch
    def __init__(self):

        super(ownCNN, self).__init__()
        self.CNN=nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=21),
            #(1200-21+2*0)/1+1 = 1180x1180 x 2    
            
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 4, kernel_size=20, stride=10),#117*117*4
            nn.ReLU(inplace=True)
            )
        self.MLP=nn.Sequential(
            nn.Linear(117, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 2),
            nn.Sigmoid()
            )
    def forward(self, x):
        x=self.CNN(x)#117*117*4
        
        F.max_pool2d(x, 4), #29*29*4
        x=self.MLP(x)
        return x
    
class CNN(nn.Module): 

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=21)
        #(1200-21+2*0)/1+1 = 1180x1180 x 2
        #590x590x2
        self.conv2 = nn.Conv2d(2, 5, kernel_size=10, stride=10)
        #590-10+2*0)/10+1=59*5
         
        #29x29x5
        
       # self.conv2_drop = nn.Dropout2d()
######
        self.fc1 = nn.Linear(4205, 1024)

        self.fc2 = nn.Linear(1024, 2)


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

  #      x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x)), 2)
        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))

        #x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return x

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs=240*240, n_outputs=2):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs,1000)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(1000,n_outputs)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output
        # self.hidden3 = Linear(n_inputs, 16384)
        # xavier_uniform_(self.hidden3.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        # X = self.hidden3(X)
        return X

# train the model
def train_model(train_dl, model, test_dl,n_epoch, checkpoints):
    # define the optimization
    checkpoint_path=checkpoints+'current_checkpoint.pt'
    best_model_path = checkpoints+"best_model.pt"
    mse_min=999
    criterion = MSELoss()
    
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    # enumerate epochs
    global T
    T=[]
    
    for epoch in range(n_epoch):
        print(epoch)
        # enumerate mini batches
        model.train()
        for i, (inputs, targets) in enumerate(tqdm(train_dl)):
            print(i)
            # clear the gradients
            optimizer.zero_grad() #13.5 us
            # compute the model output
            yhat = model(inputs) #10 ms
            # calculate loss
            loss = criterion(yhat, targets) #45 us
            # credit assignment
            
            loss.backward() # 68 ms
            
            # update model weights
            optimizer.step()
        model.eval()    
        mse=evaluate_model(test_dl, model)
        
        
        checkpoint = {
            'epoch': epoch + 1,
            'mse': mse,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        # save the model if test loss has decreased (is current best)
        if mse <= mse_min:
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            mse_min = mse
        #save model
        
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        # print(i)
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy().flatten()
        yhat = yhat.reshape((len(yhat), 1))
        actual = targets.numpy().flatten()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(traindata, model, loss_fn, optimizer)
#     test_loop(testdata, model, loss_fn)
# print("Done!")


if __name__=='__main__':
    # freeze_support()
    path=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat/"
    train=pd.read_csv(path + "trainSet.txt", index_col=None)
    test=pd.read_csv(path + "testSet.txt", index_col=None)
    
    
    encoder=LE()
    train['Prognosis']=encoder.fit_transform(train.Prognosis)
    imglabels=train[['ImageFile','Prognosis']]
    image_path_train=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images\normalizedImg/train/"
    image_path_test=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images\normalizedImg/test/"
    image_path=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images\normalizedImg/all/"

    
    imglabels_train=imglabels[imglabels.ImageFile.isin(os.listdir(image_path_train))]
    traindata=ImageDataset(imglabels_train, image_path_train)
    imglabels_test=imglabels[imglabels.ImageFile.isin(os.listdir(image_path_test))]
    testdata=ImageDataset(imglabels_test, image_path_test)

#    datapath='/hkfs/work/workspace/scratch/hw6363-project-b/RAC_train_SS_v1_full.h5'
    checkpoint_path=path + "checkpoints/"
    os.makedirs(checkpoint_path, exist_ok=True)
    # checkpoint_path='D:/'#'/hkfs/home/project/haicore-project-hzb/pt3575/'
    # datapath='D:/RAC_train_SS_v1_full.h5'
    dataset_train, dataset_test=ImageDataset(imglabels, image_path).get_splits(n_test=50)
    
    train_dl=DataLoader(dataset_train, batch_size=1, shuffle=True)
    test_dl=DataLoader(dataset_test, batch_size=1, shuffle=True)
    #model = MLP(1440000, 1)	
    
    model = ownCNN()	
    train_model(train_dl, model, test_dl, 25, checkpoint_path)
    mse = evaluate_model(test_dl, model)
    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    

# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


