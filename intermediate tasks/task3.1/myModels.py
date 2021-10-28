import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import vgg16, vgg16_bn, alexnet
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
import PIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, n_channels = 1, latent_space = 64):
        super(VAE, self).__init__()
        self.enc = Encoder(n_channels, latent_space)
        self.dec = Decoder(n_channels, latent_space)
        self.latent_space = latent_space
        
    def forward(self, x):
        mean,log_var = self.enc(x)
        reconstruction = self.dec(self.reparameterize(mean,log_var))
        return mean, log_var, reconstruction

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * log_var

    def generate_new(self, n):
        data = torch.randn(n,self.latent_space).to(device)
        return self.dec(data).detach().cpu().numpy().reshape(-1,256,256)

class Encoder(nn.Module):
    def __init__(self, n_channels, latent_space):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channels,64,3,stride = 2,padding=1),
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
          )

        self.mean_layer = nn.Sequential(
            nn.Linear(512, latent_space),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.log_var_layer = nn.Sequential(
            nn.Linear(512, latent_space),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        a = self.model(x).reshape(-1,512)
        return self.mean_layer(a), self.log_var_layer(a)

class Decoder(nn.Module):
    def __init__(self, n_channels, latent_space):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
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
            nn.Conv2d(64,n_channels,3,padding = 1),
            nn.Sigmoid(),
            )

        self.decoder_linear = nn.Sequential(
                            nn.Linear(latent_space, 512),
                            nn.LeakyReLU(),
                            nn.Dropout(0.2)
                        )

    def forward(self, x):
        return self.model(self.decoder_linear(x).view(-1,512,1,1))