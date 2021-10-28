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
    def __init__(self, train = None, get_segmentations = True, get_unlabelled = False, train_ratio = 0.75, equi_sampling = False, transform = None, drop_grades = None):
        super(MESSIDOR_256, self).__init__()
        self.transform = transform
        self.base_transform = transforms.Compose([
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
        self.equi_sampling = equi_sampling

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
                if not drop_grades is None:
                	if code in drop_grades:
                		continue
                self.image_names.append(name)
                self.image_grade.append(code)
        self.gradewise_counts = np.zeros((5))
        for i in range(5):
            self.gradewise_counts[i] = self.image_grade.count(i)
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

            if self.equi_sampling:
                self.generate_equisampling()

    def generate_equisampling(self):
        cnt = int(self.gradewise_counts.max())
        self.index_map = np.zeros((5*cnt)).astype(np.int)

        grades = np.array(self.image_grade)
        for gi in range(5):
            indices = grades == gi
            self.index_map[gi*cnt:(gi+1)*cnt] = np.random.choice(np.arange(grades.shape[0])[indices], cnt).astype(np.int)

    def __getitem__(self, i):
        if self.equi_sampling:
            index = self.index_map[i]
        else:
            index = i
        if self.image_grade[index] == -1:
            code = 'unlabelled'
        else:
            code = str(self.image_grade[index])

        im = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'images/{}'.format(code)), self.image_names[index]))
        if self.get_segmentations:
            ims = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'segments/{}'.format(code)), self.image_names[index]))
            return self.transform(im), self.base_transform(ims), self.image_grade[index]
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
        if self.equi_sampling:
            return self.index_map.shape[0]
        return len(self.image_names)


class EYEPACS_train(Dataset):
    """docstring for EYEPACS_train"""
    def __init__(self, train = None, get_segmentations = True, get_unlabelled = False, train_ratio = 0.75, equi_sampling = False, transform = None):
        super(EYEPACS_train, self).__init__()
        self.transform = transform
        self.base_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                                ])
        self.data_path = '/content/drive/Shareddrives/RND_2021_niraj_shalabh/datasets/EYEPACS/Messidor_256'
        self.image_names = []
        self.image_grade = []
        self.get_unlabelled = get_unlabelled
        self.get_segmentations = get_segmentations
        self.train = train
        self.train_ratio = train_ratio
        self.equi_sampling = equi_sampling

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
        self.gradewise_counts = np.zeros((5))
        for i in range(5):
            self.gradewise_counts[i] = self.image_grade.count(i)
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

            if self.equi_sampling:
                self.generate_equisampling()

    def generate_equisampling(self):
        cnt = int(self.gradewise_counts.max())
        self.index_map = np.zeros((5*cnt)).astype(np.int)

        grades = np.array(self.image_grade)
        for gi in range(5):
            indices = grades == gi
            self.index_map[gi*cnt:(gi+1)*cnt] = np.random.choice(np.arange(grades.shape[0])[indices], cnt).astype(np.int)

    def __getitem__(self, i):
        if self.equi_sampling:
            index = self.index_map[i]
        else:
            index = i
        if self.image_grade[index] == -1:
            code = 'unlabelled'
        else:
            code = str(self.image_grade[index])

        im = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'images/{}'.format(code)), self.image_names[index]))
        if self.get_segmentations:
            ims = PIL.Image.open(os.path.join(os.path.join(self.data_path, 'segments/{}'.format(code)), self.image_names[index]))
            return self.transform(im), self.base_transform(ims), self.image_grade[index]
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
        if self.equi_sampling:
            return self.index_map.shape[0]
        return len(self.image_names)