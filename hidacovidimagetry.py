# -*- coding: utf-8 -*-
"""HidaCovidImageTry

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EOqSFVO2HNH_7YHR0jK7SVSLEPG_xGD3
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import time
import pandas as pd
import numpy as np
from tqdm import trange, tqdm    
import matplotlib.pyplot as plt

from PIL import Image

import shutil
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder as LE
from numpy import vstack

from sklearn.metrics import mean_squared_error, accuracy_score

import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import random_split

import torch.nn as nn

from torch import Tensor
from torch.nn import Linear, Conv2d, Sequential
from torch.nn import Sigmoid,Hardsigmoid
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn.init import xavier_uniform_
from torch.nn import Softmax, ReLU, MaxPool2d
import torch.nn.functional as F

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
        image = read_image(img_path)###returns an 2D-tensor, 1200x1200
        image=image/1.0
        label = self.img_labels.iloc[idx, 1]
        label=torch.tensor(label, dtype=torch.long, device=device)
        return image, label
    
    
    def get_splits(self, n_test=50):
        test_size = n_test  # validation
        train_size = self.len - test_size  # training

        # calculate the split indexes
        return random_split(self, [train_size, test_size])
    



# train_loader=DataLoader(traindata, batch_size=16, shuffle=True)
###next step: set up model and start the fit


# device casting



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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device casting

CPUdevice = torch.device("cpu")
torch.cuda.empty_cache()
torch.manual_seed(0)



def train_model(imglabels: pd.DataFrame, image_path: str,model ,n_epoch: int, checkpoints: str):
    """performs the training of the model and evaluates the results on the test_dataset. Results are saved in the checkpoints_folder, including the accuracy per epoch""" 
    checkpoint_path=checkpoints+'current_checkpoint.pt'
    best_model_path = checkpoints+"best_model.pt"
    datei=open(checkpoints+"accuracy.txt",'w')
    datei.close()
    accuracy_max=0
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.0001, )
    dataset_train, dataset_test=ImageDataset(imglabels, image_path).get_splits(n_test=50)
    train_dl=DataLoader(dataset_train, batch_size=4, shuffle=True)
    test_dl=DataLoader(dataset_test, batch_size=4, shuffle=True)
    
    for epoch in range(n_epoch):
        #dataset_train, dataset_test=ImageDataset(imglabels, image_path).get_splits(n_test=50)
        #train_dl=DataLoader(dataset_train, batch_size=8, shuffle=True)
        #test_dl=DataLoader(dataset_test, batch_size=8, shuffle=True)
    
    
        model.train()
        preds, targs=np.array([]), np.array([])
        totalloss=0
        for i, (inputs, targets) in enumerate(tqdm(train_dl)):
            inputs, targets=inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            yhat = model(inputs)
            
            loss = criterion(yhat, targets)
            totalloss=totalloss+loss
            loss.backward()
            optimizer.step()
            #yhat=torch.max(yhat, 1)[-1].detach().cpu()
        
       #     predictions=np.concatenate((predictions,yhat))
        #    targets=np.concatenate((targets,target.detach().cpu()))   
        model.eval()   
        
        accuracy_test=evaluate_MLP_model(test_dl, model)
        accuracy_train=totalloss#evaluate_MLP_model(train_dl, model)
        datei=open(checkpoints+"accuracy.txt",'a')
        datei.write(str(epoch+1))
        datei.write("\t")
        datei.write(str(accuracy_train))
        datei.write("\t")
        datei.write(str(accuracy_test))
        datei.write("\n")
        datei.close()
        
        checkpoint = {
            'epoch': epoch + 1,
            'mse': accuracy_test,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        print(str(accuracy_test) + "///"+str(totalloss))
        #print(list(model.named_parameters())[-2][1][0])
        if accuracy_test >= accuracy_max:
        
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            accuracy_max = accuracy_test
        
        
def evaluate_MLP_model(test_dl, model):
    predictions, targets = np.array([]), np.array([])
    for i, (inputs, target) in enumerate(test_dl):
        inputs = inputs.to(device)
        
        yhat = model(inputs)
        yhat=torch.max(yhat, 1)[-1].detach().cpu()
        
        predictions=np.concatenate((predictions,yhat))
        targets=np.concatenate((targets,target.detach().cpu()))
    accuracy = accuracy_score(targets, predictions)
    return accuracy

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs,30)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        
        self.hidden2 = Linear(30,10)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        
        self.hidden3 = Linear(10, 2)
        xavier_uniform_(self.hidden3.weight)
        
    # forward propagate input
    def forward(self, X):
        X = torch.flatten(X, 1)
        X = self.hidden1(X)
        X = self.act1(X)
        
        X = self.hidden2(X)
        X = self.act2(X)
        
        X = self.hidden3(X)
       # X=Softmax(dim=1)(X)
        return X
class CNN(nn.Module): 

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=2, kernel_size=21)
        #(1200-21+2*0)/1+1 = 1180x1180 x 2
        self.actCNN1=ReLU()
        #590x590x2
        self.conv2 = Conv2d(2, 5, kernel_size=10, stride=10)
        self.actCNN2=ReLU()
        #590-10+2*0)/10+1=59*5
         
        #29x29x5
        
        # self.conv2_drop = nn.Dropout2d()
######
        self.fc1 = nn.Linear(4205, 8192)
        self.act1=Sigmoid()
        xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(8192, 1024)
        self.act2=Sigmoid()
        xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(1024, 2)
        xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x=self.conv1(x)
        x=self.actCNN1(x)
        x=MaxPool2d(2)(x)
        x=self.conv2(x)
        x=self.actCNN2(x)
        x=MaxPool2d(2)(x)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.act1(x)
        x=self.fc2(x)
        x=self.act2(x)
        x=self.fc3(x)
        
        #x=Softmax(dim=1)(x)
        return x

class CNNV2(nn.Module): 

    def __init__(self):

        super(CNNV2, self).__init__()

        self.CNNpart=nn.Sequential(
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3),
            ReLU(),
            MaxPool2d(3,3)
        )

        self.MLPpart=nn.Sequential(
             Linear(102*102, 120),
             ReLU(),
             Linear(120, 60),
             ReLU(),
             nn.Dropout(p=0.2),
             Linear(60, 2),
             Sigmoid()
        )


    def forward(self, x):
        x = self.CNNpart(x)
        x = torch.flatten(x,1)
        x = self.MLPpart(x)

        return x

path="/content/drive/MyDrive/"
#path = r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images/"
labels=pd.read_csv(path + "trainSet.txt", index_col=None)



encoder=LE()
labels['Prognosis']=encoder.fit_transform(labels.Prognosis)
imglabels=labels[['ImageFile','Prognosis']]

image_path=path+"normalizedImg/"

checkpoint_path=path + "checkpoints/"
os.makedirs(checkpoint_path, exist_ok=True)

#model = MLP(1440000, 1)	

#model = MLP(1200*1200)	
#model=CNNV2()
#model=model.cuda()
#model = Net()	
#train_model(imglabels, image_path,model, 200, checkpoint_path)
#dataset_train, dataset_test=ImageDataset(imglabels, image_path).get_splits(n_test=0)
#total_dl=DataLoader(dataset_train, batch_size=1, shuffle=True)
    
#acc = evaluate_MLP_model(total_dl, model)
#print('Accuracy: %.3f' % (acc))

dataset_train, dataset_test=ImageDataset(imglabels, image_path).get_splits(n_test=0)
total_dl=DataLoader(dataset_train, batch_size=4, shuffle=True)
bestmodel="/content/drive/MyDrive/checkpoints/best_model.pt"
model=CNNV2()
checkpoint = torch.load(bestmodel)
# initialize state_dict from checkpoint to model
model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
optimizer=Adam(model.parameters(), lr=0.0001)
optimizer.load_state_dict(checkpoint['optimizer'])
# initialize valid_loss_min from checkpoint to valid_loss_min
valid_loss_min = checkpoint['mse']



####überarbeite das DataLoading, sodass wenigstens der Patientenname mit ausgegeben wird. 
class PredictionDataset(Dataset):
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
        image = read_image(img_path)###returns an 2D-tensor, 1200x1200
        image=image/1.0
        label = self.img_labels.iloc[idx, 1]
        #label=torch.tensor(label, dtype=torch.long, device=device)
        return image, label
    
    
    def get_splits(self, n_test=50):
        test_size = n_test  # validation
        train_size = self.len - test_size  # training

        # calculate the split indexes
        return random_split(self, [train_size, test_size])

path="/content/drive/MyDrive/"
#path = r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images/"
labels=pd.read_csv(path + "trainSet.txt", index_col=None)
imglabels=labels[['ImageFile','PatientID']]
image_path=path+"normalizedImg/"
checkpoint_path=path + "checkpoints/"

#dataset_all, dataset_nix=PredictionDataset(imglabels, image_path).get_splits(n_test=0)
#total_dl=DataLoader(dataset_all, batch_size=1, shuffle=False)
#model.cuda()
#model.train()
#model.eval()
#predictions=[]
#for i, (inputs, target) in tqdm(enumerate(total_dl)):
   # inputs = inputs.to(device)
        
  #  yhat = model(inputs)
 #   yhat=yhat.detach().cpu().numpy()
        
#    predictions.append([target, yhat])

ImagePredictions_Train=pd.DataFrame([], index=np.arange(0, len(predictions)), columns=['ID','prediction_1','prediction_2'])
for idx, prediction in enumerate(predictions):
  ImagePredictions_Train.iloc[idx]=prediction[0], prediction[1].flatten()[0], prediction[1].flatten()[1]
ImagePredictions_Train.to_csv(checkpoint_path+"ImagePredictions_Train.txt", index=None)

path="/content/drive/MyDrive/"
#path = r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images/"
labels=pd.read_csv(path + "testSet_mod.txt", index_col=None)
imglabels=labels[['ImageFile','PatientID']]
image_path=path+"testSet/testSet/normalizedImg/"
checkpoint_path=path + "checkpoints/"

dataset_test, dataset_nix=PredictionDataset(imglabels, image_path).get_splits(n_test=0)
total_dl=DataLoader(dataset_test, batch_size=1, shuffle=False)
#model.cuda()
#model.train()
#model.eval()
predictions=[]
for i, (inputs, target) in tqdm(enumerate(total_dl)):
    inputs = inputs.to(device)
        
    yhat = model(inputs)
    yhat=yhat.detach().cpu().numpy()
        
    predictions.append([target, yhat])

ImagePredictions_Test=pd.DataFrame([], index=np.arange(0, len(predictions)), columns=['ID','prediction_1','prediction_2'])
for idx, prediction in enumerate(predictions):
  ImagePredictions_Test.iloc[idx]=prediction[0], prediction[1].flatten()[0], prediction[1].flatten()[1]
ImagePredictions_Test.to_csv(checkpoint_path+"ImagePredictions_Test.txt", index=None)

target