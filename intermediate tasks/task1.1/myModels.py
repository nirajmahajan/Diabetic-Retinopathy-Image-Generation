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
    def __init__(self):
        super(VAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        
    def forward(self, x):
        mean,log_var = self.enc(x)
        reconstruction = self.dec(self.reparameterize(mean,log_var))
        return mean, log_var, reconstruction

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * log_var

    def generate_new(self):
        data = torch.randn(1,2).to(device)
        return self.dec(data).detach().cpu().numpy().reshape(28,28)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(784,256, bias = True),
                nn.ReLU(inplace = True),
            )
        self.mean_layer = nn.Linear(256,2)
        self.log_var_layer = nn.Linear(256,2)

    def forward(self, x):
        a = self.model(x.reshape(-1,784))
        return self.mean_layer(a), self.log_var_layer(a)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(2,256, bias = True),
                nn.ReLU(inplace = True),
                nn.Linear(256,784),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.model(x)