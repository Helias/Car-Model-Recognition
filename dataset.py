import numpy as np
import torch
import os

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from os import path

class LocalDataset(Dataset):

    def __init__(self, base_path, txt_list, transform=None):
        self.base_path=base_path
        self.images = np.loadtxt(txt_list,dtype=str,delimiter=',')
        self.transform = transform

    def __getitem__(self, index):
        f,_,_,_,_,c = self.images[index]

        im = Image.open(path.join(self.base_path, f))

        if self.transform is not None:
            im = self.transform(im)

        label = int(c)

        return { 'image' : im, 'label':label, 'img_name': f }

    def __len__(self):
        return len(self.images)

# Algorithms to calculate mean and standard_deviation
dataset = LocalDataset("images", "training_list.csv", transform=transforms.ToTensor())

# Mean
#m = torch.zeros(3)
#for sample in dataset:
#    m += sample['image'].sum(1).sum(1)
#m /= len(dataset)*256*144

# Standard Deviation
#s = torch.zeros(3)
#for sample in dataset:
#    s+=((sample['image']-m.view(3,1,1))**2).sum(1).sum(1)
#s=torch.sqrt(s/(len(dataset)*256*144))
