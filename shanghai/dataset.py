import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import json
from torchvision import datasets, transforms



class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883


        
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target
    



def get_dataset(train_json='/home/rpu2/scratch/data/shanghai_data/part_A_train.json', test_json='/home/rpu2/scratch/data/shanghai_data/part_B_test.json', batch_size=1):
    #
    with open(train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    #
    train_loader = torch.utils.data.DataLoader(
        listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=0,
                       batch_size=batch_size,
                       num_workers=4),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
        batch_size=batch_size)   
    #
    return train_loader, test_loader



if __name__ == '__main__':
    tr, te = get_dataset()
    for i, (x,y) in enumerate(tr):
        print(f' x is {x.shape}')
        print(f' y is {y.shape}')
        break