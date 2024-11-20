from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
import random
import numpy as np


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = self.images
        #self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                #torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.ToTensor()
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                #torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        l1 = Image.open(self.labels_dir + self.labels[i])#.convert('L')

        seed = np.random.randint(0, 2 ** 31)
        # apply this seed to img tranfsorms
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.tx(i1)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.lx(l1)

        return img, label

