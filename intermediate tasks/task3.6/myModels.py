import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import vgg16, vgg16_bn, alexnet
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import ModuleList
import pickle
import matplotlib.pyplot as plt
import argparse
import sys
import os
import functools
import PIL

from iternet_parts import *

# reference https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def forward(self, x):
        return x

inv_transform = transforms.Compose([
                        transforms.ToPILImage(),
                                ])

class GANloss(nn.Module):
    """docstring for GANloss"""
    def __init__(self, gan_mode = 'lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANloss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class IternetGAN(nn.Module):
    def __init__(self, lr = 2e-4, iterations = 0):
        super(IternetGAN, self).__init__()
        self.generator = Iternet(iterations = iterations).to(device)
        self.discriminator = Discriminator().to(device)

        self.criterionGAN = GANloss().to(device)
        self.criterionL1 = torch.nn.L1Loss().to(device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=lr)
        
    def switch_off_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def generate(self, n):
        return self.generator(torch.randn(n,1,16).to(device))

    def train(self, epoch, batch_num, segmentation):
        # segmentation nx256x256
        fake_images = self.generator(torch.randn(segmentation.shape[0],1,16).to(device))

        if batch_num == 0 and (epoch % 10 == 0 or epoch < 10):
            for i in range(1):
                s = inv_transform(segmentation[i].detach().cpu())
                f = inv_transform(fake_images[i].detach().cpu())
                fig = plt.figure(figsize = (15,5))
                plt.subplot(1,2,1)
                plt.imshow(s, cmap = 'gray')
                plt.subplot(1,2,2)
                plt.imshow(f, cmap = 'gray')
                plt.savefig('images/train/epoch{}_{}.png'.format(epoch, i))


        lD = self.learn_D(segmentation.to(device), fake_images)
        lG = self.learn_G(fake_images, segmentation.to(device))
        return lG, lD

    def learn_D(self, segmentation, fake_images):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        pred_fake = self.discriminator(fake_images.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.discriminator(segmentation)
        loss_D_real = self.criterionGAN(pred_real, True)

        loss_D = (loss_D_real + loss_D_fake)/2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.item()

    def learn_G(self, fake_images, segmentation):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        pred_fake = self.discriminator(fake_images)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G = loss_G_GAN
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()



class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, input_nc = 1, ndf = 64, norm_type = 'batch'):
        super(Discriminator, self).__init__()
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            def norm_layer(x): return Identity()

        self.net = PatchGan(input_nc, ndf, n_layers=3, norm_layer=norm_layer).to(device)

    def forward(self, x):
        return self.net(x)

class PatchGan(nn.Module):
    """docstring for PatchGan"""
    def __init__(self, input_nc, ndf, n_layers, norm_layer):
        super(PatchGan, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
        

class Decoder(nn.Module):
    def __init__(self, n_channels=1, latent_space=16):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,16,3,padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,n_channels,3,padding = 1),
            nn.Sigmoid(),
            )

        self.decoder_linear = nn.Sequential(
                            nn.Linear(latent_space, 128),
                            nn.LeakyReLU(),
                            nn.Dropout(0.2)
                        )

    def forward(self, x):
        return self.model(self.decoder_linear(x).view(-1,128,1,1))

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False

        self.inc = DoubleConv(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels * 2)
        self.down2 = Down(out_channels * 2, out_channels * 4)
        self.down3 = Down(out_channels * 4, out_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(out_channels * 8, out_channels * 16 // factor)
        self.up1 = Up(out_channels * 16, out_channels * 8 // factor, bilinear)
        self.up2 = Up(out_channels * 8, out_channels * 4 // factor, bilinear)
        self.up3 = Up(out_channels * 4, out_channels * 2 // factor, bilinear)
        self.up4 = Up(out_channels * 2, out_channels, bilinear)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x1, x, logits


class MiniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=32):
        super(MiniUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False

        self.inc = DoubleConv(n_channels, out_channels)
        self.down1 = Down(out_channels, out_channels*2)
        self.down2 = Down(out_channels*2, out_channels*4)
        self.down3 = Down(out_channels*4, out_channels*8)
        self.up1 = Up(out_channels*8, out_channels*4, bilinear)
        self.up2 = Up(out_channels*4, out_channels*2, bilinear)
        self.up3 = Up(out_channels*2, out_channels, bilinear)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return x1, x, logits


class Iternet(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, out_channels=32, iterations=3):
        super(Iternet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.iterations = iterations

        self.decoder = Decoder()
        # define the network UNet layer
        self.model_unet = UNet(n_channels=n_channels,
                               n_classes=n_classes, out_channels=out_channels)

        # define the network MiniUNet layers
        self.model_miniunet = ModuleList(MiniUNet(
            n_channels=out_channels*2, n_classes=n_classes, out_channels=out_channels) for i in range(iterations))

    def forward(self, x):
        x = self.decoder(x)
        x1, x2, logits = self.model_unet(x)
        for i in range(self.iterations):
            x = torch.cat([x1, x2], dim=1)
            _, x2, logits = self.model_miniunet[i](x)

        return logits.sigmoid()