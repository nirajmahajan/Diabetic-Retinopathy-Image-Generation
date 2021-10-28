import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import vgg16, vgg16_bn, alexnet
from torch import nn, optim
from torch.nn import functional as F
import pickle
import matplotlib.pyplot as plt
import argparse
import sys
import os
import functools
import PIL

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
    def __init__(self, gan_mode = 'vanilla', target_real_label=1.0, target_fake_label=0.0):
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

class ADVAE(nn.Module):
    def __init__(self, lr = 2e-4, latent_space = 128):
        super(ADVAE, self).__init__()
        self.decoder = Decoder(1,latent_space).to(device)
        self.encoder = Encoder(1,latent_space).to(device)
        self.discriminator = Discriminator(latent_space).to(device)
        self.global_discriminator = GlobalDiscriminator().to(device)
        self.latent_space = latent_space

        self.criterionGAN = GANloss().to(device)
        self.criterionL1 = torch.nn.L1Loss().to(device)

        self.optimizer_G = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        self.optimizer_globalD = torch.optim.Adam(self.global_discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        
    def switch_off_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True
        for param in self.global_discriminator.parameters():
            param.requires_grad = True

    def generate(self, n):
        noise = torch.FloatTensor(np.random.normal(0, 1, (n, self.latent_space))).to(device)
        return self.decoder(noise)

    def generate_fake_space(self,n):
        return torch.FloatTensor(np.random.normal(0, 1, (n, self.latent_space))).to(device)

    def generate_true_space(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * log_var
        
    def train(self, e, batch_num, segmentation):
        # segmentation nx256x256
        fake_space = self.generate_fake_space(segmentation.shape[0])
        true_mu, true_sigma = self.encoder(segmentation)
        true_space = self.generate_true_space(true_mu, true_sigma)

        if batch_num == 0 and (e%10 == 0 or e < 10):
            for i in range(1):
                s = inv_transform(segmentation[i].detach().cpu())
                f = inv_transform(self.decoder(fake_space[i]).squeeze().detach().cpu())
                fig = plt.figure(figsize = (10,5))
                plt.subplot(1,2,1)
                plt.imshow(s, cmap = 'gray')
                plt.subplot(1,2,2)
                plt.imshow(f, cmap = 'gray')
                plt.savefig('images/train/epoch{}_{}.png'.format(e, i))


        lD = self.learn_D(true_space, fake_space)
        lG = self.learn_G(fake_space, segmentation)
        return lG, lD

    def learn_D(self, true_space, fake_space):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        pred_fake = self.discriminator(fake_space)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.discriminator(true_space.detach())
        loss_D_real = self.criterionGAN(pred_real, True)

        loss_D = (loss_D_real + loss_D_fake)/2
        loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_globalD.zero_grad()
        pred_decoded_fake = self.global_discriminator(self.decoder(fake_space).detach())
        loss_D_decoded_fake = self.criterionGAN(pred_decoded_fake, False)

        pred_decoded_real = self.global_discriminator(self.decoder(true_space).detach())
        loss_D_decoded_real = self.criterionGAN(pred_decoded_real, True)

        loss_D_decoded = (loss_D_decoded_real + loss_D_decoded_fake)/4
        loss_D_decoded.backward()
        self.optimizer_globalD.step()
        return loss_D.item()

    def learn_G(self, fake_space, segmentation):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        projection = self.decoder(fake_space)
        pred_fake = self.discriminator(fake_space)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        pred_decoded_fake = self.global_discriminator(projection)
        loss_G_decoded_GAN = self.criterionGAN(pred_decoded_fake, True)
        loss_G_L1 = self.criterionL1(projection, segmentation)*100
        loss_G = (loss_G_L1 + loss_G_GAN + loss_G_decoded_GAN)/2
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()



class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, latent_space = 512):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(latent_space,latent_space),
                    nn.Linear(latent_space,1),
                    nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

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
            nn.Tanh(),
            )

        self.decoder_linear = nn.Sequential(
                            nn.Linear(latent_space, 512),
                            nn.LeakyReLU(),
                            nn.Dropout(0.2)
                        )

    def forward(self, x):
        return self.model(self.decoder_linear(x).view(-1,512,1,1))

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

class GlobalDiscriminator(nn.Module):
    """docstring for GlobalDiscriminator"""
    def __init__(self, input_nc = 1, ndf = 64, norm_type = 'batch'):
        super(GlobalDiscriminator, self).__init__()
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            def norm_layer(x): return Identity()

        self.net = PatchGan(input_nc, ndf, n_layers=3, norm_layer=norm_layer).to(device)

    def forward(self, x):
        return self.net(x)