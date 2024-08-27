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
from datasets.IMDBWIKI import IMDBWIKI
from utils import AverageMeter, accuracy, adjust_learning_rate,shot_metric, shot_metric_balanced, shot_metric_cls, \
    setup_seed, balanced_metrics, soft_labeling, SoftCrossEntropy, feature_diversity, diversity_loss_regressor
#from datasets.datasets_utils import group_df
from tqdm import tqdm
# additional for focal
#from focal_loss.focal_loss import FocalLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loss import Ranked_Contrastive_Loss
from loss_contra import RnCLoss, RnCLoss_pairwise
import time
from scipy.stats import gmean
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--lds', default=False, type=bool)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='imdb_wiki',
                    choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
#parser.add_argument('--seeds', default=123, type=int, help = ' random seed ')
parser.add_argument('--tau', default=1, type=float,
                    help=' tau for logit adjustment ')
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
parser.add_argument('--aug', type=bool, default=False, help='pairwise sample contra')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)
parser.add_argument('--soft_label', type=bool, default=False)
parser.add_argument('--ce', type=bool, default=False, help='if use the cross_entropy or not')
parser.add_argument('--la', type=bool, default=False,
                    help='if use logit adj to train the imbalance')
parser.add_argument('--output_file', default='./results_', help='the output directory')
parser.add_argument('--scale', type=float, default=1,
                    help='scale of the sharpness in soft label')
parser.add_argument('--diversity', type=float, default=0, help='scale of the diversity loss')
parser.add_argument('--smooth', type=bool, default=False, help='add guassain smooth to the ce for groups')
parser.add_argument('--more_train', type=bool, default=False, help='add guassain smooth to the ce for groups')
parser.add_argument('--reg_loss', choices=['l1', 'l2'], default='l1', help='which regression loss to  use')





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
                             split='train', group_num=args.groups, group_mode=args.group_mode, reweight=args.reweight, lds = args.lds, aug=args.aug)
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


def train_one_epoch(model, train_loader, mse_loss, opt, args, e=0):
    #
    group_loss = RnCLoss(temperature=args.temp).to(device)
    #sigma, la, g_dis, gamma, ranked_contra, contra_ratio, soft_label, ce = \
    #    args.sigma, args.la, args.g_dis, args.gamma, args.ranked_contra, args.contra_ratio, args.soft_label, args.ce
    for idx, (x, y, g, w) in enumerate(train_loader):
        
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    #
    store_names = f'Supervised_group_lr_{args.lr}_epoch_{args.epoch}_bs_{args.batch_size}_groups_{args.groups}.pth'
    #
    if args.soft_label:
        store_names = 'soft_label_' + store_names
    #
    print(" store name is ", store_names)
    print(" time is  ", time.asctime())
    #
    store_name =  store_names + '.txt'
    #
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_dataset(
        args)
    #
    if args.reg_loss == 'l2':
        loss_reg = nn.MSELoss()
    if args.reg_loss == 'l1':
        loss_reg = nn.L1Loss()
    else:
        print(f' no regression loss special')
    #
    model = ResNet_regression(args).to(device)
    #
    model_val = ResNet_regression(args).to(device)
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #adjust_learning_rate(opt, e, args)
        model, tol = train_one_epoch(
            model, train_loader, loss_reg, opt, args, e)
        if e % 20 == 0 or e == (args.epoch - 1):
            cls_acc, reg_mae,  mean_L1_pred,  mean_L1_gt, shot_dict_val_pred, shot_dict_val_pred_gt = validate(
                model, val_loader, train_labels, e)
            #
            write_val_log('./output/' + store_name, cls_acc, reg_mae,  mean_L1_pred,
                          mean_L1_gt, shot_dict_val_pred, shot_dict_val_pred_gt, tol)
            # add the validation to train
            # new line
            if args.more_train:
                model, tol = train_one_epoch(model, val_loader, loss_ce, loss_mse, opt, args, e)
            #
            if best_bMAE > mean_L1_pred and e > 40:
                best_bMAE = mean_L1_pred
                torch.save(model.state_dict(),
                           './models/model_{}.pth'.format(store_names))
    #load the best model
    model_val.load_state_dict(torch.load(
        './models/model_{}.pth'.format(store_names)))
    #
    acc_gt, acc_pred, g_pred, mae_gt, mae_pred, shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred, _ = \
        test_step(model_val, test_loader, train_labels, args)
    print(' Val model mse of gt is {}, mse of pred is {}, acc of the group assignment is {}, \
            mae of gt is {}, mae of pred is {} to_avg is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred, np.mean(tole)))


    

   
