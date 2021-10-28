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

class VAE_GAN(nn.Module):
    def __init__(self, lr = 2e-4, latent_space = 32):
        super(VAE_GAN, self).__init__()
        self.decoder = Decoder(1,latent_space).to(device)
        self.encoder = Encoder(1,latent_space).to(device)
        self.discriminator_code = Discriminator_code().to(device)
        self.latent_space = latent_space

        self.criterionGAN1 = GANloss('vanilla').to(device)
        self.criterionL1 = torch.nn.L1Loss().to(device)

        #self.optimizer_G = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()), lr=lr, betas=(.5, 0.999))
        #self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self.criterionGAN2 = GANloss('lsgan').to(device)
        self.optimizer_G = torch.optim.Adam(list(self.generator.parameters())+list(self.decoder.parameters())+list(self.encoder.parameters()), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.discriminator.parameters())+list(self.discriminator_code.parameters()), lr=lr, betas=(.5, 0.999))


    def switch_off_discriminator(self):
            for param in self.discriminator.parameters():
                param.requires_grad = False

            for param in self.discriminator_code.parameters():
                param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

        for param in self.discriminator_code.parameters():
            param.requires_grad = True

    def generate(self, n):
        noise = torch.FloatTensor(np.random.normal(0, 1, (n, self.latent_space))).to(device)
        return self.generator(self.decoder(noise).to(device))
        #return self.decoder(noise)

    def generate_true_space(self,n):
        return torch.FloatTensor(np.random.normal(0, 1, (n, self.latent_space))).to(device)

    def generate_fake_space(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps * log_var
        
    def train(self, e, batch_num, segmentation, retinopathy):
        # segmentation nx256x256
        true_space = self.generate_true_space(segmentation.shape[0])
        fake_mu, fake_sigma = self.encoder(segmentation)
        fake_space = self.generate_fake_space(fake_mu, fake_sigma)
        #fake_images = self.generator(self.decoder(fake_space))

        if batch_num == 0 and (e%10 == 0 or e < 10):
            for i in range(1):
                s = inv_transform(segmentation[i].detach().cpu())
                f = inv_transform(self.decoder(fake_space[i]).squeeze().detach().cpu())
                #r = inv_transform(fake_images[i].detach().cpu())
                fig = plt.figure(figsize = (10,5))
                plt.subplot(1,2,1)
                plt.imshow(s, cmap = 'gray')
                plt.subplot(1,2,2)
                plt.imshow(f, cmap = 'gray')
                #plt.subplot(1,3,3)
                #plt.imshow(r)
                plt.savefig('images/train/epoch{}_{}.png'.format(e, i))


        lD = self.learn_D(true_space, fake_space, segmentation, retinopathy)
        lG = self.learn_G(fake_space, segmentation, retinopathy)
        return lG, lD

    def learn_D(self, true_space, fake_space, segmentation, retinopathy):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        pred_fake = self.discriminator_code(fake_space.detach())
        loss_D_fake = self.criterionGAN1(pred_fake, False)

        pred_real = self.discriminator_code(true_space)
        loss_D_real = self.criterionGAN1(pred_real, True)

        fake_seg_gen = self.decoder(fake_space)
        fake_reti = self.generator(fake_seg_gen)

        fakepairs = torch.cat((fake_seg_gen.detach(), fake_reti.detach()), 1)
        pred_fake2 = self.discriminator(fakepairs)
        loss_D_fake2 = self.criterionGAN2(pred_fake2, False)

        realpairs = torch.cat((segmentation, retinopathy), 1)
        pred_real2 = self.discriminator(realpairs)
        loss_D_real2 = self.criterionGAN2(pred_real2, True)

        loss_D = (loss_D_real + loss_D_fake + loss_D_real2 + loss_D_fake2)/2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.item()

    def learn_G(self, fake_space, segmentation, retinopathy):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        pred_fake = self.discriminator_code(fake_space)
        loss_G_GAN = self.criterionGAN1(pred_fake, True)
        loss_G_L1 = self.criterionL1(self.decoder(fake_space), segmentation)*100
        
        fake_seg_gen = self.decoder(fake_space)
        fake_reti = self.generator(fake_seg_gen)

        fakepairs = torch.cat((fake_seg_gen, fake_reti), 1)
        pred_fake2 = self.discriminator(fakepairs)
        loss_G_GAN2 = self.criterionGAN2(pred_fake2,True)

        loss_G_L1_2 = self.criterionL1(fake_reti, retinopathy)*100
        
        
        
        loss_G = loss_G_L1 + loss_G_GAN + loss_G_GAN2 + loss_G_L1_2
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()



class Discriminator_code(nn.Module):
    """docstring for Discriminator"""
    def __init__(self):
        super(Discriminator_code, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(32,64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64,1),
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
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
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
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
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



"""GAN MODEL"""

'''
class Pix2pix(nn.Module):
    def __init__(self, lr = 2e-4):
        super(Pix2pix, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.criterionGAN = GANloss().to(device)
        self.criterionL1 = torch.nn.L1Loss().to(device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        
    def switch_off_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def generate(self, segmentation):
        return self.generator(segmentation.to(device))

    def train(self, epoch, batch_num, segmentation, retinopathy):
        # segmentation nx256x256
        # retinopathy nx256x256
        fake_images = self.generator(segmentation.to(device))

        if batch_num == 0 and (epoch % 10 == 0 or epoch < 10):
            for i in range(1):
                r = inv_transform(retinopathy[i].detach().cpu())
                s = inv_transform(segmentation[i].detach().cpu())
                f = inv_transform(fake_images[i].detach().cpu())
                fig = plt.figure(figsize = (15,5))
                plt.subplot(1,3,1)
                plt.imshow(r)
                plt.subplot(1,3,2)
                plt.imshow(s, cmap = 'gray')
                plt.subplot(1,3,3)
                plt.imshow(f)
                plt.savefig('images/train/epoch{}_{}.png'.format(epoch, i))


        lD = self.learn_D(segmentation.to(device), fake_images, retinopathy.to(device))
        lG = self.learn_G(fake_images, segmentation.to(device), retinopathy.to(device))
        return lG, lD

    def learn_D(self, segmentation, fake_images, retinopathy):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        fakepairs = torch.cat((segmentation, fake_images.detach()), 1)
        pred_fake = self.discriminator(fakepairs)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        realpairs = torch.cat((segmentation, retinopathy), 1)
        pred_real = self.discriminator(realpairs)
        loss_D_real = self.criterionGAN(pred_fake, True)

        loss_D = (loss_D_real + loss_D_fake)/2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.item()

    def learn_G(self, fake_images, segmentation, retinopathy):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        fakepairs = torch.cat((segmentation, fake_images.detach()), 1)
        pred_fake = self.discriminator(fakepairs)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_L1 = self.criterionL1(fake_images, retinopathy)*100
        loss_G = loss_G_L1 + loss_G_GAN
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()
'''


class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, input_nc = 4, ndf = 64, norm_type = 'batch'):
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
        

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self, input_nc = 1, output_nc = 3, ngf = 64, norm_type = 'batch', use_dropout = True):
        super(Generator, self).__init__()
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'none':
            def norm_layer(x): return Identity()

        self.net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer, use_dropout=use_dropout).to(device)

    def forward(self, x):
        return self.net(x)


class UnetGenerator(nn.Module):
    """docstring for UnetGenerator"""
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout = False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)