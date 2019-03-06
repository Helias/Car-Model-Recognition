import numpy as np
import torch
import os

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from os import path
from glob import glob

from config import *

dirs = glob(IMAGES_PATH + "/*/")

num_classes = {}

i = 0
for d in dirs:
    d = d.replace(IMAGES_PATH, "")
    d = d.replace("/", "")
    num_classes[d] = i
    i+=1

def get_class(idx):
    for key in num_classes:
        if idx == num_classes[key]:
            return key

def preprocessing():
    train_csv = ""
    test_csv  = ""
    class_files_training = []
    class_files_testing  = []

    for key in num_classes:
        class_files = glob(IMAGES_PATH+"/"+str(key)+"/*")
        class_files = [w.replace(IMAGES_PATH+"/"+str(key)+"/", "") for w in class_files]
        class_files.sort()

        class_files_training = class_files[: int(len(class_files)*.66)] # get 66% class images fo training
        class_files_testing = class_files[int(len(class_files)*.66)+1 :] # get 33% class images fo training

        for f in class_files_training:
            if "," in f:
                os.rename(IMAGES_PATH+"/"+key+"/"+f, IMAGES_PATH+"/"+key+"/"+f.replace(",",""))
                f = f.replace(",", "")
            train_csv += f + ","+str(key)+"\n"

        for f in class_files_testing:
            if "," in f:
                os.rename(IMAGES_PATH+"/"+key+"/"+f, IMAGES_PATH+"/"+key+"/"+f.replace(",",""))
                f = f.replace(",", "")
            test_csv += f + ","+str(key)+"\n"

    train_csv_file = open("train_file.csv", "w+")
    train_csv_file.write(train_csv)
    train_csv_file.close()

    test_csv_file = open("test_file.csv", "w+")
    test_csv_file.write(test_csv)
    test_csv_file.close()
#preprocessing()


class LocalDataset(Dataset):

    def __init__(self, base_path, txt_list, transform=None):
        self.base_path=base_path
        self.images = np.loadtxt(txt_list,dtype=str,delimiter=',')
        self.transform = transform

    def __getitem__(self, index):
        f,c = self.images[index]

        image_path = path.join(self.base_path + "/" + str(c), f)
        im = Image.open(image_path)

        if self.transform is not None:
            im = self.transform(im)

        label = num_classes[c]

        return { 'image' : im, 'label':label, 'img_name': f }

    def __len__(self):
        return len(self.images)
