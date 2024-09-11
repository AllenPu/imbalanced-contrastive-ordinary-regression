import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from utils import get_lds_kernel_window, shot_count, check_shot
import math
import torch
from PIL import ImageFilter 
import random


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight='none', group_num=10, max_age=100, aug=False):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.group_range = int(max_age/group_num)
        self.group_list = []
        self.group_num = group_num
        #
        self.aug = aug
        self.elr_index_return = False
        #
        if self.split == 'train':
            # set up the three shot 
            maj_shot, med_shot, min_shot = shot_count(self.df['age'])
            #three_shots = {}
            #
            group_dic = {x: 0 for x in range(group_num)}
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                age = min(row['age'], 100)
                #
                shot_key = check_shot(age, maj_shot, med_shot, min_shot)
                #three_shots[shot_key] = three_shots.get(shot_key, 0) + 1
                #
                group_id = math.floor(age/self.group_range)
                group_dic[min(group_id, group_num-1)] += 1
            list_group = sorted(group_dic.items(),
                                key=lambda group_dic: group_dic[0])
            # return how many number of samples in a group
            self.group_list = [i[1] for i in list_group]
            #self.three_shots = three_shots
        else:
            pass


    def __len__(self):
        return len(self.df)

    def get_group_list(self):
        return self.group_list

    # add to test the multi expert
    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(
            self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        if self.aug:
            transform2 = self.aug_transform()
            img1, img2 = transform(img).unsqueeze(0), transform2(img).unsqueeze(0)
            imgs = torch.cat((img1, img2), dim=0)
            #print(f' size  {img.shape}')
            # shape : bsz, 2,  3， 244， 244
        else:
            imgs = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        group_ = min(math.floor(label/self.group_range), self.group_num-1)
        group = np.asarray([group_]).astype('float32')
        #print(f' size  {img.shape}')
        if self.elr_index_return:
            return index, imgs, label, group
        else:
            return imgs, label, group
        

        
    def enable_multi_crop(self, enable=False):
        if enable:
            self.aug=True

    
    #def enable_elr_index_return(self, enable=False):
    #    if enable:
    #        self.elr_index_return=True



    #def get_three_shots_num_list(self):
    #    l = []
    #    for i in  ['maj', 'med', 'min']:
    #        l.append(self.three_shots[i])
    #    return l

    

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform


    # return additioanl transform
    def aug_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
            ])
        return train_transform
    
    

    def _prepare_weights(self, reweight, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            # clip weights for inverse re-weight
            value_dict = {k: np.clip(v, 5, 1000)
                          for k, v in value_dict.items()}
        num_per_label = [
            value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(
                lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [
                smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
    






class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

