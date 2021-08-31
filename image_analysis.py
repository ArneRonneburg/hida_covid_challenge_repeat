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
path=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat/"
train=pd.read_csv(path + "trainSet.txt", index_col=None)
test=pd.read_csv(path + "testSet.txt", index_col=None)


encoder=LE()
train['Prognosis']=encoder.fit_transform(train.Prognosis)
imglabels=train[['ImageFile','Prognosis']]
from torchvision.io import read_image
from torch.utils.data import Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



image_path=r"C:\Users\Arne\Documents\DataScience\hida_covid_challenge_repeat\images\normalizedImg/"
traindata=CustomImageDataset(imglabels, image_path)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
train_loader=DataLoader(traindata, batch_size=16, shuffle=True)
###next step: set up model and start the fit

import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

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







# imagelist=os.listdir(image_path)
# names=[]
# images=np.zeros((len(imagelist),1440000))
# for j in range(0, len(imagelist)):
#     i=imagelist[j]
#     print(i)
    
#     im=np.asarray(PIL.Image.open(image_path+i)).flatten()
#     names.append(i)
#     images[j]=im
#     #images=pd.concat((images, pd.DataFrame(im)), ignore_index=True)
# Images=pd.DataFrame(images)    
# prognosis=[]
# for i in range(0, len(names)):
#     p=np.array(train[train.ImageFile==names[i]].Prognosis)[0]
#     if p=="MILD":
#         prognosis.append(0)
#     else:
#         prognosis.append(1)