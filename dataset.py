'''
Author: your name
Date: 2020-11-21 20:28:56
LastEditTime: 2020-11-21 23:52:29
LastEditors: Liu Chen
Description: 
FilePath: \BACN\dataset.py
  
'''
import os
import json
import random
import pandas as pd
import numpy as np
from os.path import join as opj
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageEnhance, ImageFilter
import skimage
from skimage import util

# modify the config.py in different enviornment for path configuration
from config import *
training_dataset_root = TRAIN_DATA_ROOT
train_gt = TRAIN_GT_PATH


# training data
class Train_Collection(Dataset):
    def __init__(self, img_size=(64, 64)):
        sourceA = opj(training_dataset_root, 'patch')
        sourceB = opj(training_dataset_root, 'seps', '18')
        dists_path = opj(training_dataset_root, 'dists_forms')
        gts = json.load(open(train_gt))
        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)

    def __len__(self):
        return len(self.A_imgs)

    def __getitem__(self, item):
        pn = random.randint(0, 1)
        if pn == 0:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]

            imgB = Image.open(self.B_imgs[item])
            imgBname = os.path.split(self.B_imgs[item])[1]
            label = 1
        # negative condition
        else:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            while True:
                choice = random.randint(0, len(self.B_imgs)-1)
                if choice != item:
                    break
            imgB = Image.open(self.B_imgs[choice])
            imgBname = os.path.split(self.B_imgs[choice])[1]
            label = 0

        label = torch.Tensor([label])
        img_A = self.trans(imgA)
        img_B = self.trans(imgB)

        return img_A, img_B, label

class Direct_data(Train_Collection):
    def __init__(self):
        super(Direct_data, self).__init__()
    def __getitem__(self, i):
        a,b,l = super(Direct_data, self).__getitem__(i)
        x = torch.cat([a.flatten(),b.flatten()],0)
        return x, l.squeeze()


ver_dataset_root = VER_DATA_ROOT
ver_gt = VER_GT_PATH


class Verific_Collection(Dataset):
    def __init__(self, img_size=(64, 64)):
        sourceA = opj(ver_dataset_root, 'patch')
        sourceB = opj(ver_dataset_root, 'seps', '18')
        gts = json.load(open(ver_gt))

        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)

    def __len__(self):
        return len(self.A_imgs)

    def __getitem__(self, item):
        pn = random.randint(0, 1)
        if pn == 0:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            imgB = Image.open(self.B_imgs[item])
            imgBname = os.path.split(self.B_imgs[item])[1]
            label = 1
        # negative condition
        else:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            while True:
                choice = random.randint(0, len(self.B_imgs)-1)
                if choice != item:
                    break
            imgB = Image.open(self.B_imgs[choice])
            imgBname = os.path.split(self.B_imgs[choice])[1]
            label = 0

        label = torch.Tensor([label])
        img_A = self.trans(imgA)
        img_B = self.trans(imgB)

        return img_A, img_B, label


class Search_collection850(Dataset):
    def __init__(self, img_size=(64, 64), noisy=False):
        self.noisy = noisy
        # VER_GT_PATH = r'/home/liuchen/BACN/data/test_gt.json'
        sourceA = opj(ver_dataset_root, 'patch')
        sourceB = opj(ver_dataset_root, 'seps', '18')
        gts = json.load(open(ver_gt))

        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)

        tmp_list = [self.trans(Image.open(img)).unsqueeze(0) 
                    for img in self.B_imgs]
        self.B_Tensors = torch.cat(tmp_list, dim=0)
        self.labelmap = [int(os.path.split(img)[1].split('.')[0]) 
                        for img in self.B_imgs]

    def __len__(self):
        return len(self.A_imgs)

    def __getitem__(self, item):
        org_imgA = Image.open(self.A_imgs[item])
        if self.noisy:
            org_imgA = self.data_augment(org_imgA)
        A_Tensor = self.trans(org_imgA).squeeze().unsqueeze(0)
        

        A_Tensor=A_Tensor.expand_as(self.B_Tensors)
        A_label = int(os.path.split(self.A_imgs[item])[1].split('.')[0])
        return A_Tensor, self.B_Tensors, A_label, self.labelmap
    
    def data_augment(self, x):
        """ 
        rotation, shift, scale change 
        imput: Image.open() 
        """
        # # rotate and crop center
        # p = random.uniform(0,1)
        # if p>1.0:
        #     angle = random.randint(-180,180)
        #     x = x.rotate(angle)
        #     # x = x.crop([x.size[0]/4,x.size[1]/4,x.size[0]*3/4,x.size[1]*3/4])
        
        # Brigtness change
        brightness = random.uniform(0.5, 1)
        bri = ImageEnhance.Brightness(x)
        x = bri.enhance(factor=brightness)
        # Contract change
        contra = random.uniform(0.5, 2)
        enh_con = ImageEnhance.Contrast(x)
        x = enh_con.enhance(factor=contra)
        
        # Blurring
        p = random.uniform(-1,1)
        if p > 0:
            x = x.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Gaussian Noise
        p = random.uniform(-1,1)
        if p > 0:
            f = np.array(x)
            noise_gs_img = util.random_noise(f,mode='gaussian', var=0.5)
            x = Image.fromarray(np.uint8(noise_gs_img*255))
        
        # # random crop
        # rdcrop_sizefactor = (random.uniform(0.95,1.0))
        # rdcropsize = tuple([int(rdcrop_sizefactor * x.size[0]), 
        #                  int(rdcrop_sizefactor * x.size[1])])
        # x = transforms.RandomCrop(rdcropsize)(x)
        # transforms.CenterCrop
        # ctcrop_sizefactor = 0.5
        # ctcropsize = tuple([int(ctcrop_sizefactor * x.size[0]), 
        #                  int(ctcrop_sizefactor * x.size[1])])
        # x = transforms.CenterCrop(ctcropsize)(x)
        return x

if __name__ == '__main__':
    a=Search_collection850()
    print(a.__len__())