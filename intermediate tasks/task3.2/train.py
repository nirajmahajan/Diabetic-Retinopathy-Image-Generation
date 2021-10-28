import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from myModels import AdvEncoder,AdvDecoder,AdVAE,Discriminator_VAE
from myDatasets import MESSIDOR_256


adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = AdvEncoder()
decoder = AdvDecoder()
discriminator = Discriminator_VAE()
cuda = True if torch.cuda.is_available() else False

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 32
n_epochs = 200
TRAIN_BATCH_SIZE = 1

# Configure data loader
#testset = MESSIDOR_256(train = False, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
#testloader = torch.utils.data.DataLoader(testset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2)
trainset = MESSIDOR_256(train = True, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for i, (_,segs, _) in enumerate(trainloader):

        # Adversarial ground truths
        valid = Variable(Tensor(segs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(segs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(segs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        mu,log_var = encoder(real_imgs)
        #mu,log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z_dash = torch.randn_like(std)
        encoded_imgs = mu + std*z_dash
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (segs.shape[0],latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch,n_epochs, i, len(trainloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(trainloader) + i
        if batches_done % 400 == 0:
            sample_image(n_row=10, batches_done=batches_done)
