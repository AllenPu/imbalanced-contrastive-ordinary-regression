from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import os
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import pickle
import random
from PIL import ImageFilter


class IMDBWIKI(data.Dataset):
    def __init__(self, df, data_dir, img_size = 224, split='train', group_num = 10, lds = False, group_mode = 'i_g', aug = False, reweight = None, max_group=100):
        self.groups = group_num
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split    
        self.group_range = max_group/group_num
        self.group_mode = group_mode
        self.aug = aug
        self.re_weight = reweight
        self.lds = lds
        #self.key_list = [i for i in range(group_num)]
        # key is the group is, value is the group num
        #
        if split == 'train':
            group_dict = {}
            bin_dict = {}
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                age = row['age']
                group_id = math.floor(age/self.group_range)
                # put the age 0 into the first group
                if group_id > self.groups - 1:
                    group_id = self.groups - 1
                else:
                    group_dict[group_id] = group_dict.get(group_id, 0) + 1
                #
                bin_dict[age] = bin_dict.get(age, 0) + 1
            #
            list_group = sorted(group_dict.items(), key = lambda group_dict : group_dict[0])
            self.group_list = [i[1] for i in list_group]
            #
            # calculate the number of each age to construct a balanced group, if not in train set, set 0
            if self.group_mode == 'b_g':
                #for i in range(101):
                #    if i not in bin_dict.keys():
                #        bin_dict[i] = 0
                self.list_bin = sorted(bin_dict.items(), key= lambda bin_dict : bin_dict[0])
                #print(f" list bin {self.list_bin} length {len(self.list_bin)} type {type(self.list_bin)}")
                self.bin_list = [j[1] for j in self.list_bin]
                _, _, self.mapping = self.eq_groups(self.groups)
                with open('./mapping.pkl', 'wb') as f:
                    pickle.dump(self.mapping, f)
            #
            #with open('bin_dict_smooth.pkl', 'wb') as f:
            #    pickle.dump(bin_dict, f)
            
            #
            self.weights = self.weights_prepare(reweight=reweight, lds=lds)
        else:
            if self.group_mode == 'b_g':
                with open('./mapping.pkl', 'rb') as f:
                    self.mapping = pickle.load(f)
        #else:
        #    pass
        
    def __len__(self):
        return len(self.df)
    


    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        #
        transform = self.get_transform()
        if not self.aug:
            img = transform(img)
        else :
            transform2 = self.aug_transform()
            img1, img2 = transform(img).unsqueeze(0), transform2(img).unsqueeze(0)
            img = torch.cat((img1, img2), dim=0)
        label = np.asarray([row['age']]).astype('float32')
        # imbalanced each group
        if self.group_mode  == 'i_g':
            group_index = math.floor(label/self.group_range)
            if group_index > self.groups - 1:
                group_index = self.groups - 1
            group = np.asarray([group_index]).astype('float32')
        # balanced each group 
        elif self.group_mode == 'b_g':
            group_id = self.mapping[row['age']]
            group = np.asarray([group_id]).astype('float32')
        else:
            print(" group mode should be defined! ")
        # ordinary binary label with [1,0] denotes 1, [0,1] denotes 0
        if self.split == 'train':
            if self.re_weight is not None or self.lds is True:
                weight = np.asarray([self.weights[index]]).astype(
                    'float32') if self.weights is not None else np.asarray([np.float32(1.)])
                return img, label, group, weight
            else:
                return img, label, group, 1
        elif self.split == 'val':
            return img, label, group, 1
        else:
            return img, label, group


    def get_group(self):
        return self.group_list




    def enable_multi_crop(self, enable=False):
        if enable:
            self.aug=True




    def get_transform(self):
        if self.split == 'train' and not self.aug :
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        elif self.split == 'train' and self.aug :
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                #transforms.RandomApply([
                #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                #], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform
    
    # return additional transform
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


    def weights_prepare(self, reweight='sqrt_inv', max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {None, 'inverse', 'sqrt_inv'}
        #
        value_dict = {x: 0 for x in range(max_target)}
        #
        labels = self.df['age'].values
        #
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
        if not len(num_per_label):
            print(" None num_per_label ")
            return None
        if reweight is not None:
            print(f"Using re-weighting: [{reweight.upper()}]")
        if lds:
            lds_kernel_window = self.get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        #print(f" weights is {weights}")

        if reweight is None and lds is False:
            return None
        else:
            return weights




    def get_lds_kernel_window(self, kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window
    


    def eq_groups(self, classes=10):
        #print(self.bin_list)
        N = sum(self.bin_list)
        print(f'N is {N}')
        new_class = {}
        new_class_bin = {}
        mapping = {}
        cum = 0
        for i in range(len(self.bin_list)):
            cum += self.bin_list[i]
            index = max(round(classes/N * cum)-1, 0)
            new_class[index] = new_class.get(index, 0) + self.bin_list[i]
            mapping[self.list_bin[i][0]] = index
            if type(new_class_bin.get(index, 0)) is list:
                new_class_bin[index].append(i)
            else:
                new_class_bin[index] = [i]
        #print(f' mapping is {mapping}')
        #print(f' group is {self.group_list}')
        return new_class, new_class_bin, mapping
    

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



 





