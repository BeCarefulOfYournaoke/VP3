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
import random
from datetime import datetime

#导入相关模块
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from PIL import Image
import torch, os
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import Callable, Optional
    
class ReadTwo(Dataset): #继承Dataset
    def __init__(self, root_dir, transform: Optional[Callable] = None): #__init__是初始化该类的一些基础参数
        self.root_dir = root_dir   #文件目录
        self.transform = transform #变换
        self.coarseClasses = os.listdir(self.root_dir)#所有类名
        self.coarseClasses.sort(key=lambda x:(x[0].zfill(2) if (x[0].isdigit() and x[1].isalpha()) else x[:2]))
        self.fineClasses = []
        self.coarseClassPath = []
        self.imgsPath = []
        self.labels = []
        self.flag = 0
        #针对Cifar20，子父类
        for i in self.coarseClasses:
            self.coarseClassPath.append(os.path.join(self.root_dir, i))
            for j in os.listdir(os.path.join(self.root_dir, i)):
                self.fineClasses.append(os.path.join(self.root_dir, i, j))
                for k in os.listdir(os.path.join(self.root_dir, i, j)):
                    self.imgsPath.append(os.path.join(self.root_dir, i, j, k))
                    self.labels.append(self.flag)
            self.flag = self.flag + 1
        
        for path in self.imgsPath[:10]:
            print(f'img path is :{path}')

        self.imgsPath = np.array(self.imgsPath)
        self.labels = np.array(self.labels)
        self.maxlabel = np.max(self.labels)
        minlabel = np.min(self.labels)
        self.names = self.__dict__
        for i in range(self.maxlabel+1):
            print(f'class {i}: {self.coarseClasses[i]}')
            self.names['label'+str(i)] = np.argwhere(self.labels==i)
        #print(self.imgsPath)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):#根据索引index返回dataset[index]
        image_path = self.imgsPath[index]#根据索引index获取该图片
        label = self.labels[index]
        #加0是因为返回的是[x,],后边还有一个空维度！！！
        
        anchor = Image.open(image_path).convert('RGB')
    
        if self.transform:
            anchor = self.transform(anchor)
        label = torch.tensor(label, dtype=torch.long)
        return anchor, label, image_path
