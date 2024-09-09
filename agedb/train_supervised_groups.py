import argparse
from symbol import parameters
import numpy as np
import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from loss import LAloss
from network import ResNet_regression
from datasets.agedb import AgeDB
#from datasets.datasets_utils import group_df
from tqdm import tqdm
# additional for focal
#from focal_loss.focal_loss import FocalLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loss import Ranked_Contrastive_Loss, RnCLoss_pairwise
import time
from scipy.stats import gmean
from utils import setup_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--lds', default=False, type=bool)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
                    #default='/home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--group_mode', default='i_g', type=str,
                    help=' b_g is balanced group mode while i_g is imbalanced group mode')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
#parser.add_argument('--regulize', type=bool, default=False, help='if to regulaize the previous classification results')
parser.add_argument('--fl', type=bool, default=False,
                    help='if use focal loss to train the imbalance')
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
parser.add_argument('--g_dis', type=bool, default=False,
                    help='if dynamically adjust the tradeoff')
parser.add_argument('--gamma', type=float, default=5, help='tradeoff rate')
parser.add_argument('--reweight', type=str, default=None,
                    help='weight : inv or sqrt_inv')
parser.add_argument('--ranked_contra', type=bool, default=False, help='group  wise contrastive')
parser.add_argument('--aug', type=bool, default=True, help='pairwise sample contra')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)
parser.add_argument('--output_file', default='./results_', help='the output directory')





def get_dataset(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                          split='train', reweight=args.reweight, group_num=args.groups, aug=args.aug)
    #
    group_list = train_dataset.get_group_list()
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                         img_size=args.img_size, split='test', group_num=args.groups)
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, group_list, train_labels


def train_contrastive_epoch(model, train_loader, opt, args):
    #
    group_loss = RnCLoss_pairwise(temperature=args.temp).to(device)
    #sigma, la, g_dis, gamma, ranked_contra, contra_ratio, soft_label, ce = \
    #    args.sigma, args.la, args.g_dis, args.gamma, args.ranked_contra, args.contra_ratio, args.soft_label, args.ce
    for idx, (x, _, g) in enumerate(train_loader):
        x, g = x.to(device), g.to(device)
        bsz = x.shape[0]
        if args.aug:
            x = x.reshape(-1,x.shape[-3], x.shape[-2], x.shape[-1])
        #print(f'x shape is {x.shape}, g shape is {g.shape}')
        _, z = model(x)
        z = z.reshape(bsz,2,-1)
        #split into two parts : first is the group, second is the prediction
        #y_chunk = torch.chunk(y_output, 2, dim=1)
        loss = group_loss(z, g)
        loss.backward()
        opt.step()
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    #
    store_names = f'Supervised_group_lr_{args.lr}_epoch_{args.epoch}_bs_{args.batch_size}_groups_{args.groups}'
    #
    print(" store name is ", store_names)
    print(" time is  ", time.asctime())
    #
    store_name =  store_names + '.txt'
    #
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_dataset(
        args)
    #
    model = ResNet_regression(args).to(device)
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    for e in tqdm(range(args.epoch)):
        model = train_contrastive_epoch(model, train_loader, opt, args)
    data_set = args.dataset
    torch.save(model, f'./checkpoint/{data_set}_{store_names}.pth')





    

   
