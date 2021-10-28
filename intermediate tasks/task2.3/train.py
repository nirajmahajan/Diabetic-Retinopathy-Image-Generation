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
import seaborn as sn
import pandas as pd
import progressbar
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import gc
from tqdm import tqdm

from myDatasets import EYEPACS_train_64

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
parser.add_argument('--load_pretrained', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--id', type = str, default = '1.1')
args = parser.parse_args()

seed_torch(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 60
lr = 3e-2
SAVE_INTERVAL = 10
NUM_EPOCHS = 200
STEP_SIZE = 20
LR_GAMMA = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EXPERIMENT_ID = args.id
expected_id = '2.3'
assert(expected_id == EXPERIMENT_ID)
losses = []
accuracies = []
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

train_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomResizedCrop(256, scale=(0.9, 0.98)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.6193246715450339, 0.5676388422333433, 0.5303413730576545), (0.12337693906775953, 0.09914381078783173, 0.06671092824144163)),
                                transforms.RandomErasing(
                                                    p=0.20,
                                                    scale=(0.03, 0.08),
                                                    ratio=(0.5, 2.0)
                                                )
                                        ])
test_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.6193246715450339, 0.5676388422333433, 0.5303413730576545), (0.12337693906775953, 0.09914381078783173, 0.06671092824144163)),
                                        ])

testset = EYEPACS_train_64(train = False, get_segmentations = False, get_unlabelled = False, train_ratio = 0.75, equi_sampling = False, transform = test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2)
trainset = EYEPACS_train_64(train = True, get_segmentations = False, get_unlabelled = False, train_ratio = 0.75, equi_sampling = True, transform = train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)
traintestset = EYEPACS_train_64(train = True, get_segmentations = False, get_unlabelled = False, train_ratio = 0.75, equi_sampling = False, transform = test_transform)
traintestloader = torch.utils.data.DataLoader(traintestset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = False)

model = models.resnet152(pretrained = True).to(device)
num_ftrs = model.fc.in_features
out_ftrs = 5
model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),nn.ReLU(),nn.Linear(512,out_ftrs)).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if args.load_pretrained:
    d = torch.load(os.path.join(runtime_path, 'pretrained.pth'))
    model.load_state_dict(d['model_state_dict'])

for name,child in model.named_children():
    if name in ['layer2','layer3','layer4','fc']:
        #print(name + 'is unfrozen')
        for param in child.parameters():
            param.requires_grad = True
    else:
        #print(name + 'is frozen')
        for param in child.parameters():
            param.requires_grad = False
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_GAMMA)

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
    optimizer.load_state_dict(dic['optimizer'])
    scheduler.load_state_dict(dic['scheduler'])
    losses = dic['losses']
    print('Resuming Training at {} from epoch {}'.format(pre_e,EXPERIMENT_ID))
else:
    model_state = 0
    pre_e =0
    print('Starting Training at {}'.format(EXPERIMENT_ID))

def is_eval_mode():
    return args.eval

def train(e):
    tot_loss = 0
    tot_correct = 0

    for batch_num,(images,grades) in tqdm(enumerate(trainloader), desc = 'Training for epoch {}'.format(e), total = len(trainloader)):
        optimizer.zero_grad()
        preds = model(images.to(device))
        l = criterion(preds, grades.to(device))
        l.backward()
        optimizer.step()
        tot_loss += l
        tot_correct += (grades.cpu().numpy() == preds.argmax(1).detach().cpu().numpy()).sum()
    scheduler.step()
    print('Total train Loss for epoch = {}'.format(100*tot_loss/batch_num))
    print('Train Accuracy = {}'.format(100*tot_correct/len(trainset)))
    return tot_loss/batch_num

def validate(train = False, silent = True):
    tot_loss = 0
    tot_correct = 0
    conf_mat = np.zeros((5,5))
    if train:
        if not silent:
            print('\nEvaluating on the trainset')
        dloader = traintestloader
        dset = traintestset
    else:
        if not silent:
            print('\nEvaluating on the testset')
        dloader = testloader
        dset = testset

    with torch.no_grad():
        if not silent:
            bar = progressbar.ProgressBar(maxval=len(dloader), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            bari = 0

        for batch_num,(images,grades) in enumerate(dloader):
            preds = model(images.to(device))
            l = criterion(preds, grades.to(device))
            tot_loss += l
            if not silent:
                bari += 1
                bar.update(bari)
            g = grades.cpu().numpy()
            p = preds.argmax(1).cpu().numpy()
            tot_correct += (g == p).sum()
            for i,li in enumerate(g.astype(np.int)):
                conf_mat[li, p[i]] += 1
    if not silent:
        bar.finish()
        print('Total CE Loss = {}'.format(tot_loss/batch_num))
    df_cm = pd.DataFrame(conf_mat, range(5), range(5))
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    if train:
      plt.savefig('conf_train.png')
    else:
      plt.savefig('conf_test.png')

    if train:
        print('Train Accuracy = {}'.format(100*tot_correct/len(dset)))
    else:
        print('Test Accuracy = {}'.format(100*tot_correct/len(dset)))
    return tot_correct/len(dset)
    

if args.eval:
    validate(train = True, silent = True)
    validate(train = False, silent = True)
    os._exit(0)

for e in range(NUM_EPOCHS):

    model_state = e//SAVE_INTERVAL
    if pre_e > 0:
        pre_e -= 1
        continue

    if e % SAVE_INTERVAL == 0:
        seed_torch(args.seed)

    l = train(e)
    losses.append(l)
    a = validate(train = False, silent = True)
    accuracies.append(a)

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimizer'] = optimizer.state_dict()
    dic['scheduler'] = scheduler.state_dict()
    dic['losses'] = losses
    dic['accuracies'] = accuracies


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
