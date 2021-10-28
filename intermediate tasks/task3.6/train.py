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

from myModels import IternetGAN
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
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--id', type = str, default = '1.1')
args = parser.parse_args()

train_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                        ])

seed_torch(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 10
lr = 5e-4
SAVE_INTERVAL = 25
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 5000
EXPERIMENT_ID = args.id
expected_id = '3.6'
assert(expected_id == EXPERIMENT_ID)
glosses = []
dlosses = []
torch.autograd.set_detect_anomaly(True)


runtime_path = '/content/drive/Shareddrives/RND_2021_niraj_shalabh/tasks/{}/'.format(EXPERIMENT_ID)
if not os.path.isdir(runtime_path):
    os.mkdir(runtime_path)
local_path = './models/'
if not os.path.isdir(local_path):
    os.mkdir(local_path)
if not os.path.isdir('./images/'):
    os.mkdir('./images')
    os.mkdir('./images/train')
    os.mkdir('./images/test')

trainset = MESSIDOR_256(train = None, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75, transform = train_transform, drop_grades = [3,4])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)

model = IternetGAN(iterations = 0)

with open('force_save.txt', 'w') as f:
    f.write('0')

global pre_e
pre_e = 0

if args.resume or args.resume_from_drive:
    if args.resume_from_drive:
        print('Copying the checkpoint to the runtime')
        os.system("cp -r '{}'state.pth ./models".format(runtime_path))
        model_state = torch.load(local_path + 'state.pth')['state']
        os.system("cp -r '{}'checkpoint_{}.pth ./models".format(runtime_path,model_state))
    model_state = torch.load(local_path + 'state.pth')['state']
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
    print('Resuming Training at {} from epoch {}'.format(pre_e,EXPERIMENT_ID))
else:
    model_state = 0
    pre_e =0
    print('Starting Training at {}'.format(EXPERIMENT_ID))

def is_eval_mode():
    return args.eval

def train(e):
    print('\nTraining for epoch {}'.format(e))
    tot_loss_g = 0
    tot_loss_d = 0

    bar = progressbar.ProgressBar(maxval=len(trainloader), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bari = 0

    for batch_num,(_,segments,grades) in enumerate(trainloader):
        lg, ld = model.train(e, batch_num, segments.to(device))
        tot_loss_d += ld
        tot_loss_g += lg
        bari += 1
        bar.update(bari)

    bar.finish()
    print('Total Generator Loss for epoch = {}'.format(tot_loss_g/batch_num))
    print('Total Discriminator Loss for epoch = {}'.format(tot_loss_d/batch_num))
    return tot_loss_g/batch_num, tot_loss_d/batch_num

def validate():
    print('\nTesting')
    with torch.no_grad():
        fake = model.generate(20).cpu().numpy()
        for i in range(fake.shape[0]):
            f = fake[i]
            fig = plt.figure(figsize = (15,5))
            plt.imshow(f[0], cmap = 'gray')
            plt.savefig('images/test/test_{}.png'.format(i))


if args.eval:
    validate()
    os._exit(0)

for e in range(NUM_EPOCHS):

    model_state = e//SAVE_INTERVAL
    if pre_e > 0:
        pre_e -= 1
        continue

    if e % SAVE_INTERVAL == 0:
        seed_torch(args.seed)

    lg, ld = train(e)
    glosses.append(lg)
    dlosses.append(ld)

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimizer_G'] = model.optimizer_G.state_dict()
    dic['optimizer_D'] = model.optimizer_D.state_dict()
    dic['dlosses'] = dlosses
    dic['glosses'] = glosses


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, local_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, local_path + 'state.pth')
        print('Saving model to {}'.format(runtime_path))
        print('Copying checkpoint to drive')
        os.system("cp -r ./models/checkpoint_{}.pth '{}'".format(model_state, runtime_path))
        os.system("cp -r ./models/state.pth '{}'".format(runtime_path))
        os.system("cp -r ./images '{}'".format(runtime_path))

    forced = False
    with open('force_save.txt', 'r') as f:
        a = f.read().strip()
        if a == '1':
            print('Force copying checkpoint to drive')
            torch.save(dic, local_path + 'checkpoint_{}.pth'.format(model_state))
            torch.save({'state': model_state}, local_path + 'state.pth')
            print('Saving model to {}'.format(runtime_path))
            os.system("cp -r ./models/checkpoint_{}.pth '{}'".format(model_state, runtime_path))
            os.system("cp -r ./models/state.pth '{}'".format(runtime_path))            
            os.system("cp -r ./images '{}'".format(runtime_path))            
            forced = True
    if forced:
        with open('force_save.txt', 'w') as f:
            f.write('0')
