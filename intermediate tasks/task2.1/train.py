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
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--id', type = str, default = '1.1')
args = parser.parse_args()

seed_torch(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 30
lr = 3e-3
SAVE_INTERVAL = 25
NUM_EPOCHS = 2000
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EXPERIMENT_ID = args.id
expected_id = '2.1'
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
if not os.path.isdir(os.path.join(runtime_path,'./images/')):
    os.mkdir(os.path.join(runtime_path,'./images/'))
    os.mkdir(os.path.join(runtime_path,'./images/train'))
    os.mkdir(os.path.join(runtime_path,'./images/test'))

testset = MESSIDOR_256(train = False, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
testloader = torch.utils.data.DataLoader(testset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2)
trainset = MESSIDOR_256(train = True, get_segmentations = True, get_unlabelled = True, train_ratio = 0.75)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)

model = Pix2pix()

with open('force_save.txt', 'w') as f:
    f.write('0')

global pre_e
pre_e = 0

if args.resume or args.resume_from_drive:
    if args.resume_from_drive:
        print('Copying the checkpoint to the runtime')
        os.system("cp -r '{}'state.pth* ./models".format(runtime_path))
        model_state = torch.load(local_path + 'state.pth')['state']
        os.system("cp -r '{}'checkpoint_{}.pth ./models".format(runtime_path, model_state))
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

    for batch_num,(images,segments,grades) in enumerate(trainloader):
        lg, ld = model.train(e, batch_num, segments, images)
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
    
    bar = progressbar.ProgressBar(maxval=len(trainloader), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bari = 0

    inv_transform = transforms.Compose([
                        transforms.ToPILImage(),
                                ])
    with torch.no_grad():
        for batch_num,(images,segments,grades) in enumerate(testloader):
            fake = model.generate(segments.to(device)).cpu()
            for i in range(images.shape[0]):
                num = batch_num*TRAIN_BATCH_SIZE + i
                r = inv_transform(images[i].detach().cpu())
                s = segments[i].squeeze().numpy()
                f = inv_transform(fake[i].detach().cpu())
                fig = plt.figure(figsize = (15,5))
                plt.subplot(1,3,1)
                plt.imshow(r)
                plt.subplot(1,3,2)
                plt.imshow(s, cmap = 'gray')
                plt.subplot(1,3,3)
                plt.imshow(f)
                plt.savefig('images/test/test_{}.png'.format(num))
            bari += 1
            bar.update(bari)

    os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
    os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))
    bar.finish()
    

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
        os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
        os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))

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
            os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
            os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))
            forced = True
    if forced:
        with open('force_save.txt', 'w') as f:
            f.write('0')
