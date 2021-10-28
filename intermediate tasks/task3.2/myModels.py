import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import progressbar
import time
from tqdm import tqdm
#!pip install pytorch-model-summary
from pytorch_model_summary import summary
import os
import copy

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'


from torchvision import models

class AdvEncoder(nn.Module):
    def __init__(self):
        super(AdvEncoder, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(1,64,3,stride = 2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,3,stride = 2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,3,stride = 2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,3,stride = 2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,2),
            nn.BatchNorm2d(512))
        
        self.fc_mu = nn.Linear(512,32)
        self.fc_logvar = nn.Linear(512,32)

    def forward(self,x):      #x = b,1,256,256
        y = self.feature1(x)  #y = b,32,1,1
        #print(y.shape)
        y = torch.flatten(y,start_dim=1) #y = b,32
        #print(y.shape)
        mu = self.fc_mu(y)    #mu = b,32
        logvar = self.fc_logvar(y)  #sig = b,32
        return mu,logvar;



class AdvDecoder(nn.Module):
    def __init__(self):
        super(AdvDecoder, self).__init__()
        self.fc1 = nn.Linear(32,512)

        self.feature2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,1,3,padding = 1),
            )
        
        

    def forward(self,x):                #x = b,32
        y = self.fc1(x)                 #x = b,512
        y = torch.unsqueeze(y,2)
        y = torch.unsqueeze(y,3)        #y = b,512,1,1 
        y = self.feature2(y)
        print(y.shape)
        recons_im = F.sigmoid(y)    #recons_im = b,1,256,256
        return recons_im;        


class AdVAE(nn.Module):
    def __init__(self):
        super(AdVAE,self).__init__()
        self.encode = AdvEncoder()
        self.decode = AdvDecoder()

    def forward(self,x):
        mu,log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z_dash = torch.randn_like(std)
        z = mu + std*z_dash         #z = b,32
        im_vessel = self.decode(z)
        return im_vessel,z


class Discriminator_VAE(nn.Module):
    def __init__(self):
        super(Discriminator_VAE,self).__init__()
        self.fc1 = nn.Linear(32,64)
        self.fc2 = nn.Linear(64,1)

    def forward(self,z):
        y = self.fc2(self.fc1(z))
        y = F.sigmoid(y)
        return y;


print(summary(AdVAE(), torch.zeros((10,1,256,256)), show_input=False))
