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

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        
    def forward(self, x, co):
        mean,log_var = self.enc(x,co)

        reconstruction = self.dec(self.reparameterize(mean,log_var),co)
        return mean, log_var, reconstruction

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*log_var

    def generate_new(self,co):
        data = torch.randn(1,2).to(device)
        c = torch.tensor([co]).to(device)
        return self.dec(data,c).detach().cpu().numpy().reshape(28,28)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(794,256, bias = True),
                nn.ReLU(inplace = True),
            )
        self.mean_layer = nn.Linear(256,2)
        self.log_var_layer = nn.Linear(256,2)

    def forward(self, x, co):
        c = idx2onehot(co, n=10)
        x = torch.cat((x, c), dim=-1)
        a = self.model(x.reshape(-1,794))
        return self.mean_layer(a), self.log_var_layer(a)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(12,256, bias = True),
                nn.ReLU(inplace = True),
                nn.Linear(256,784),
                nn.Sigmoid(),
            )

    def forward(self, x, c0):
        c = idx2onehot(c0, n=10)
        x = torch.cat((x, c), dim=-1)
        return self.model(x)