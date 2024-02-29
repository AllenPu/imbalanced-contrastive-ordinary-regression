import argparse
from symbol import parameters
import numpy as np
import torchvision
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
import argparse
import pandas as pd
from datasets.IMDBWIKI import IMDBWIKI
from utils import *
#from datasets.datasets_utils import group_df
from tqdm import tqdm
# additional for focal
#from focal_loss.focal_loss import FocalLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loss import Ranked_Contrastive_Loss
from loss_contra import RnCLoss
import time
from scipy.stats import gmean
from models import *
from loss_contra import *
from collections import OrderedDict
from train import test_step, write_test_loggs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='imdb_wiki',
                    choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--tau', default=1, type=float,
                    help=' tau for logit adjustment ')
parser.add_argument('--group_mode', default='i_g', type=str,
                    help=' b_g is balanced group mode while i_g is imbalanced group mode')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
#parser.add_argument('--regulize', type=bool, default=False, help='if to regulaize the previous classification results')
parser.add_argument('--la', action='store_true',
                    help='if use logit adj to train the imbalance')
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
parser.add_argument('--ranked_contra', action='store_true')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)
parser.add_argument('--soft_label', action='store_true')
parser.add_argument('--ce', action='store_true',  help='if use the cross_entropy /la or not')
parser.add_argument('--aug', action='store_true')
parser.add_argument('--aug_model', action='store_true')
parser.add_argument('--epoch_cls', default=80,type=int)
parser.add_argument('--epoch_reg', default=0, type=int)
parser.add_argument('--hybird_epoch', default=0, type=int)
parser.add_argument('--output_file', default='./results_rnc', help='the output directory')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')






def get_dataset(args):
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    #if args.group_mode == 'b_g':
    #    nb_groups = int(args.groups)
    #    df = group_df(df, nb_groups)
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    ##### how to orgnize the datastes
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                             split='train', group_num=args.groups, group_mode=args.group_mode, reweight=args.reweight, aug=args.aug)
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val, img_size=args.img_size,
                           split='val', group_num=args.groups, group_mode=args.group_mode)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test, img_size=args.img_size,
                            split='test', group_num=args.groups, group_mode=args.group_mode)
    #
    train_group_cls_num = train_dataset.get_group()
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
    #
    train_labels = df_train['age']
    #
    return train_loader, test_loader, val_loader, train_group_cls_num, train_labels



def train_encoder_one_epoch(model, optimizer, e, criterion, losses, args):
    #
    for idx, (x, y, g, _) in enumerate(train_loader):
        #
        bsz = y.shape[0]
        #adjust_learning_rate(args, optimizer, e)
        y, g = y.to(device), g.to(device)
        optimizer.zero_grad()
        #
        if args.aug:
            images = torch.cat([x[0], x[1]], dim=0).to(device)
            align = y
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        else:
            images = x.to(device)
            align = g
            features = model(images)
        #features = model(images)
        #f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, align)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, losses
            



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (0.1 ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epoch)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, args, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        #'epoch': epoch,
    }
    torch.save(state, save_file)
    del state




def train_epoch(model, train_loader, opt, args):
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()
    
    for idx, (x, y, g, _) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        opt.zero_grad()
        y_output, z = model(x)
        #
        y_ =  torch.chunk(y_output,2,dim=-1)
        g_hat, y_hat = y_[0], y_[1]
        y_pred = torch.gather(y_hat, dim=1, index=g.to(torch.int64)) 
        #
        if args.soft_label:
            g_soft_label = soft_labeling(g, args).to(device)
                #print(f' g hat is {g_hat[:8]} soft label {g_soft_label[:8]}')
                #assert 1== 2
            loss_ce = SoftCrossEntropy(g_hat, g_soft_label)
                #print(f' soft label loss is {loss_ce.item()}')
        if args.ce:
            loss_ce = F.cross_entropy(g_hat, g.squeeze().long(), reduction='mean')
                #print(f' ce loss is {loss_ce.item()}')
            #if torch.isnan(loss_ce):
            #    print(f' g_hat is {g_hat[:10]} g is {g[:10]} z is {z[:10]}')
            #    assert 1==0
        loss_mse = mse(y_pred, y)
            #print(f' mse is {loss_mse.item()}, ce is {loss_ce.item()}')
        loss = loss_mse + loss_ce
        loss.backward()
        opt.step()
    print(f' mse is {loss_mse.item()}, ce is {loss_ce.item()}')
    return model




def get_model(args):
    model = Encoder_regression(groups=args.groups, name='resnet50')
    # load pretrained
    if args.aug_model:
        ckpt = torch.load('ckpt_aug_sample.pth')
    else:
        ckpt = torch.load('ckpt_aug_group.pth')
    #
    #ckpt = torch.load('last.pth')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys]=v
    model.encoder.load_state_dict(new_state_dict)
    # freeze the pretrained part
    for (name, param) in model.encoder.named_parameters():
        param.requires_grad = False
    #
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer




if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    store_name = args.output_file + '.txt'
    #####
    #
    train_loader, test_loader, val_loader, train_group_cls_num, train_labels = get_dataset(args)
    #
    #model = Encoder('resnet50').to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=1e-4)
    model, optimizer = get_model(args)
    #if args.aug:
    #    criterion = RnCLoss_pairwise(temperature=args.temp, label_diff='l1', feature_sim='l2')
    #else:
    #    criterion = RnCLoss(temperature=args.temp, label_diff='l1', feature_sim='l2')
    #
    for e in tqdm(range(args.epoch)):
        acc_gt, acc_pred, g_pred, mae_gt, mae_pred, shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred, group_and_pred = test_step(model, train_loader, train_labels, args)
        break
        model = train_epoch(model, train_loader, optimizer, args)
        acc_gt, acc_pred, g_pred, mae_gt, mae_pred, shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred, group_and_pred = \
            test_step(model, test_loader, train_labels, args)
        results_test = [acc_gt, acc_pred, g_pred, mae_gt, mae_pred, gmean_gt, gmean_pred ]
    #write_test_loggs('./output/'+store_name, results_test, shot_dict_pred,
    #            shot_dict_gt, shot_dict_cls, args)
        if e%5 == 0 or e == args.epoch-1:
            print(' current epoch is {}'.format(e))
            print(' mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
                    mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred)+"\n")
            #
            print(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_pred['many']['l1'],
                                                                             shot_dict_pred['median']['l1'], shot_dict_pred['low']['l1']) + "\n")
            #
            print(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_gt['many']['l1'],
                                                                     shot_dict_gt['median']['l1'], shot_dict_gt['low']['l1']) + "\n")
            #
            print(' CLS Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_cls['many']['cls'],
                                                                         shot_dict_cls['median']['cls'], shot_dict_cls['low']['cls']) + "\n")
            #
            print(' G-mean Gt {}, Many :  G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_gt, shot_dict_gt['many']['gmean'],
                                                                         shot_dict_gt['median']['gmean'], shot_dict_gt['low']['gmean'])+ "\n")                                                       
            #
            print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_dict_pred['many']['gmean'],
                                                                         shot_dict_pred['median']['gmean'], shot_dict_pred['low']['gmean'])+ "\n")                                                       
        #


        #model, losses = train_encoder_one_epoch(model, optimizer, e, criterion, losses, args)
        #if e%20 == 0:
            #print(f' In epoch {e} losses is {losses.avg}')
            #save_model(model, optimizer, args, save_file= f'ckpt_aug_{args.aug}.pth')
    


    

    