import numpy as np
import torch
import random
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
import PIL
import time
import progressbar
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import gc

from myModels import Pix2pix
from myModels2 import ADVAE
from myDatasets import MESSIDOR_256

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--resume_from_drive', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
args = parser.parse_args()

seed_torch(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 30
lr = 3e-3
SAVE_INTERVAL = 25
NUM_EPOCHS = 2000
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EXPERIMENT_ID = '2.1'
glosses = []
dlosses = []
torch.autograd.set_detect_anomaly(True)


runtime_path = '/content/drive/Shareddrives/RND_2021_niraj_shalabh/tasks/{}/'.format(EXPERIMENT_ID)
runtime_path2 = '/content/drive/Shareddrives/RND_2021_niraj_shalabh/tasks/{}/'.format('3.3.3')
if not os.path.isdir(runtime_path):
    os.mkdir(runtime_path)
local_path = './models/'
local_path2 = './models2/'
if not os.path.isdir(local_path):
    os.mkdir(local_path)
if not os.path.isdir(local_path2):
    os.mkdir(local_path2)
if not os.path.isdir('./images/'):
    os.mkdir('./images')
    os.mkdir('./images/train')
    os.mkdir('./images/test')
    os.mkdir('./images/test2')
if not os.path.isdir(os.path.join(runtime_path,'./images/')):
    os.mkdir(os.path.join(runtime_path,'./images/'))
    os.mkdir(os.path.join(runtime_path,'./images/train'))
    os.mkdir(os.path.join(runtime_path,'./images/test2'))

testset = MESSIDOR_256(train = False, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
testloader = torch.utils.data.DataLoader(testset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2)
trainset = MESSIDOR_256(train = True, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)

model = Pix2pix()
model2 = ADVAE()

global pre_e
pre_e = 0

if args.resume_from_drive:
    print('Copying the checkpoints to the runtime')
    os.system("cp -r '{}'state.pth* ./models".format(runtime_path))
    model_state = torch.load(local_path + 'state.pth')['state']
    os.system("cp -r '{}'checkpoint_{}.pth ./models".format(runtime_path, model_state))

    os.system("cp -r '{}'state.pth* ./models2".format(runtime_path2))
    model_state2 = torch.load(local_path2 + 'state.pth')['state']
    os.system("cp -r '{}'checkpoint_{}.pth ./models2".format(runtime_path2, model_state2))
model_state = torch.load(local_path + 'state.pth')['state']
model_state2 = torch.load(local_path2 + 'state.pth')['state']
if (not args.state == -1):
    model_state = args.state
print('Loading checkpoint at model state {}'.format(model_state))
dic = torch.load(local_path + 'checkpoint_{}.pth'.format(model_state))
pre_e = dic['e']
model.load_state_dict(dic['model'])
model.optimizer_G.load_state_dict(dic['optimizer_G'])
model.optimizer_D.load_state_dict(dic['optimizer_D'])
glosses = dic['glosses']
dlosses = dic['dlosses']
dic2 = torch.load(local_path2 + 'checkpoint_{}.pth'.format(model_state2))
pre_e2 = dic2['e']
model2.load_state_dict(dic2['model'])
model2.optimizer_G.load_state_dict(dic2['optimizer_G'])
model2.optimizer_D.load_state_dict(dic2['optimizer_D'])
glosses2 = dic2['glosses']
dlosses2 = dic2['dlosses']

def validate():
    print('\nTesting')
    
    inv_transform = transforms.Compose([
                        transforms.ToPILImage(),
                                ])
    with torch.no_grad():
        model2.eval()
        fakesegments = model2.generate(20).cpu()
        model.eval()
        fake = model.generate(fakesegments.to(device)).cpu()
        for i in range(fake.shape[0]):
            s = fakesegments[i].squeeze().numpy()
            f = inv_transform(fake[i].detach().cpu())
            fig = plt.figure(figsize = (10,5))
            plt.subplot(1,2,1)
            plt.imshow(s, cmap = 'gray')
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.savefig('images/test2/test_{}.png'.format(i))

    os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
    os.system("cp -r ./images/test2/* '{}'".format(os.path.join(runtime_path,'./images/test2/')))
    

validate()
