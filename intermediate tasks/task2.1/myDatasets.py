import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import os
import PIL

class MESSIDOR_256(Dataset):
    """docstring for MESSIDOR_256"""
    def __init__(self, train = None, get_segmentations = True, get_unlabelled = False, train_ratio = 0.75):
        super(MESSIDOR_256, self).__init__()
        self.transform = train_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                                ])
        self.data_path = '/content/drive/Shareddrives/RND_2021_niraj_shalabh/datasets/MESSIDOR1/Messidor_256'
        self.image_names = []
        self.image_grade = []
        self.get_unlabelled = get_unlabelled
        self.get_segmentations = get_segmentations
        self.train = train
        self.train_ratio = train_ratio

        for root, dirs, files in os.walk(os.path.join(self.data_path, 'images'), topdown=True):
            for name in sorted(files):
                path = os.path.join(root, name)
                code = path.split('/')[-2]
                if code == 'unlabelled':
                    if self.get_unlabelled:
                        code = -1
                    else:
                        continue
                else:
                    code = int(code)
                self.image_names.append(name)
                self.image_grade.append(code)
        if not self.train is None:
            cnts = {}
            tmpcnts = {}
            for code in range(-1,5):
                cnts[code] = self.image_grade.count(code)
                tmpcnts[code] = 0
            bcknames = self.image_names.copy()
            bckgrades = self.image_grade.copy()
            self.image_names.clear()
            self.image_grade.clear()

            for n, g in zip(bcknames, bckgrades):
                if train:
                    tmpcnts[g] += 1
                    if tmpcnts[g] > int(cnts[g]*self.train_ratio):
                        continue
                    self.image_names.append(n)
                    self.image_grade.append(g)
                else:
                    tmpcnts[g] += 1
                    if tmpcnts[g] <= int(cnts[g]*self.train_ratio):
                        continue
                    self.image_names.append(n)
                    self.image_grade.append(g)

    def __getitem__(self, index):
        if self.image_grade[index] == -1:
            code = 'unlabelled'
        else:
            code = str(self.image_grade[index])
        im = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'images/{}'.format(code)), self.image_names[index]))
        if self.get_segmentations:
            ims = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'segments/{}'.format(code)), self.image_names[index]))
            return self.transform(im), self.transform(ims), self.image_grade[index]
        else:
            return self.transform(im), self.image_grade[index]

    def display(self, index):
        if self.image_grade[index] == -1:
            code = 'unlabelled'
        else:
            code = str(self.image_grade[index])
        im = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'images/{}'.format(code)), self.image_names[index]))
        ims = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'segments/{}'.format(code)), self.image_names[index]))
        fig = plt.figure()
        plt.suptitle('Retinopathy grade: {}'.format(code))
        plt.subplot(1,2,1)
        plt.imshow(im)
        plt.subplot(1,2,2)
        plt.imshow(ims, cmap = 'gray')
        plt.show()



    def __len__(self):
        return len(self.image_names)