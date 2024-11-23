import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from PIL import Image
import torch, os
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class MySubset(Dataset):
    def __init__(self, D, indices, transform=None):
        self.transform = None if transform else D.transform
        self.imgs = list(map(lambda x:D.imgs.__getitem__(x)[0],indices))
        self.targets = list(map(lambda x:D.targets.__getitem__(x),indices))
    def __getitem__(self,idx):
        return self.transform(Image.open(self.imgs[idx])), self.targets[idx], self.imgs[idx]
    def __len__(self): return len(self.targets)
    
class MyDataSet(Dataset): 
    def __init__(self, root_dir, transform=None): 
        self.root_dir = root_dir   
        self.transform = transform 
        self.classes = os.listdir(self.root_dir)
        self.classPath = []
        self.imgsPath = []
        self.labels = []
        self.flag = 0
        for i in self.classes:
            self.classPath.append(os.path.join(self.root_dir, i))
        for i in self.classPath:
            imgsname = os.listdir(i)
            for j in imgsname:
                self.imgsPath.append(os.path.join(i, j))
                self.labels.append(self.flag)
            self.flag = self.flag + 1
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        image_path = self.imgsPath[index]
        img = Image.open(image_path).convert('RGB')
        
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        
        return img, label, image_path
