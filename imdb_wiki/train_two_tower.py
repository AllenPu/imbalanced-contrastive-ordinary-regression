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
from loss import LAloss
from network import ResNet_two_tower
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
from loss_contra import RnCLoss
import time
from scipy.stats import gmean


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
parser.add_argument('--ce', action='store_false',  help='if use the cross_entropy /la or not')
parser.add_argument('--epoch_cls', default=80)
parser.add_argument('--epoch_reg', default=40)




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
                             split='train', group_num=args.groups, group_mode=args.group_mode, reweight=args.reweight)
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



def train_one_epoch(model, train_loader, args, opt=[], mode = 'cls'):
    [opt_extractor, opt_cls, opt_reg] = opt
    ce_loss = F.cross_entropy()
    mse_loss = F.mse_loss()
    if mode == 'cls':
        model = freeze_module(model, mode)
    elif mode == 'reg':
        model = freeze_module(model, mode)
    #
    for idex, ( x, y, g) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        opt_extractor.zero_grad()
        opt_cls.zero_grad()
        opt_reg.zero_grad()
        #
        if mode == 'cls':
            g_hat, z = model(x, 'cls')
            loss = ce_loss(g_hat, g.squeeze().long())
            loss.backward()
            opt_extractor.step()
            opt_cls.step()
        if mode == 'reg':
            g_hat, y_hat, z = model(x, 'reg')
            y_predicted = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
            loss = mse_loss(y_predicted, y)
            loss.backward()
            #opt_extractor.step()
            opt_reg.step()
    return model
            
def test_step(model, test_loader, train_labels, args):
    model.eval()
    mse_gt = AverageMeter()
    #mse_mean = AverageMeter()
    acc_g = AverageMeter()
    acc_mae_gt = AverageMeter()
    mse_pred = AverageMeter()
    acc_mae_pred = AverageMeter()
    mse = nn.MSELoss()
    # gmean
    criterion_gmean_gt = nn.L1Loss(reduction='none')
    criterion_gmean_pred = nn.L1Loss(reduction='none')
    gmean_loss_all_gt, gmean_loss_all_pred = [], [] 
    # this is for y
    pred_gt, pred, labels = [], [], []
    # CHECK THE PREDICTION ACC
    pred_g_gt, pred_g = [], []
    #
    with torch.no_grad():
        for idx, (inputs, targets, group) in enumerate(test_loader):
            #
            bsz = targets.shape[0]
            #
            inputs = inputs.to(device)
            targets = targets.to(device)
            group = group.to(device)
            # for regression
            labels.extend(targets.data.cpu().numpy())
            # for cls, cls for g
            pred_g_gt.extend(group.data.cpu().numpy())
            #
            g_hat, y_hat, z = model(inputs.to(torch.float32) , 'reg')
            #
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            #
            group = group.to(torch.int64)
            #
            y_gt = torch.gather(y_hat, dim=1, index=group)
            y_pred = torch.gather(y_hat, dim=1, index=g_index)
            #  the regression results for y
            pred.extend(y_pred.data.cpu().numpy())
            pred_gt.extend(y_gt.data.cpu().numpy())
            # the cls results for g
            pred_g.extend(g_index.data.cpu().numpy())
            #
            mse_y_gt = mse(y_gt, targets)
            mse_y_pred = mse(y_pred, targets)
            #
            mae_loss_gt = torch.mean(torch.abs(y_gt - targets))
            #
            mae_loss_pred = torch.mean(torch.abs(y_pred - targets))
            #
            acc3 = accuracy(g_hat, group, topk=(1,))
            # gmean
            loss_all_gt = criterion_gmean_gt(y_gt, targets)
            loss_all_pred = criterion_gmean_pred(y_pred, targets)
            gmean_loss_all_gt.extend(loss_all_gt.cpu().numpy())
            gmean_loss_all_pred.extend(loss_all_pred.cpu().numpy())
            #
            mse_gt.update(mse_y_gt.item(), bsz)
            #mse_mean.update(mse_mean_1.item(), bsz)
            mse_pred.update(mse_y_pred.item(), bsz)
            acc_g.update(acc3[0].item(), bsz)
            acc_mae_gt.update(mae_loss_gt.item(), bsz)
            acc_mae_pred.update(mae_loss_pred.item(), bsz)
        # gmean
        gmean_gt = gmean(np.hstack(gmean_loss_all_gt), axis=None).astype(float)
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        # shot metric for predictions
        shot_dict_pred = shot_metric(pred, labels, train_labels)
        shot_dict_gt = shot_metric(pred_gt, labels, train_labels)
        #
        shot_dict_cls = shot_metric_cls(pred_g, pred_g_gt, train_labels,  labels)
        return mse_gt.avg,  mse_pred.avg, acc_g.avg, acc_mae_gt.avg, acc_mae_pred.avg,\
                                    shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred


def freeze_module(model, model_name='cls'):
    if model_name=='cls':
        for name, param in model.model_extractor.named_parameters():
            param.requires_grad = True
        for name, param in model.model_cls.named_parameters():
            param.requires_grad = False
        for name, param in model.model_reg.named_parameters():
            param.requires_grad = True
    elif model_name=='reg':
        for name, param in model.model_extractor.named_parameters():
            param.requires_grad = False
        for name, param in model.model_cls.named_parameters():
            param.requires_grad = True
        for name, param in model.model_reg.named_parameters():
            param.requires_grad = False
    else:
        print(" Invalid module name !!!")
    return model




if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    #####
    #
    train_loader, test_loader, val_loader, train_group_cls_num, train_labels = get_dataset(args)
    #
    model = ResNet_two_tower(args)
    opt_extractor, opt_cls, opt_reg = model.setup_opt(args)
    opts = [opt_extractor, opt_cls, opt_reg]
    #
    for e in tqdm(range(args.epoch_cls)):
        model = train_one_epoch(model, train_loader, args, opts, 'cls')
    for e in tqdm(range(args.epoch_reg)):
        model = train_one_epoch(model, train_loader, args, opts, 'reg')
    #
    mse_gt,  mse_pred, acc_g, acc_mae_gt, acc_mae_pred, shot_dict_pred, shot_dict_gt, \
        shot_dict_cls, gmean_gt, gmean_pred = test_step(model, test_loader, args, train_labels, args)
    #
    print(f'mse_gt is {mse_gt}, mse_pred is {mse_pred}, acc_g is {acc_g}, acc_mae_gt is {acc_mae_gt}, acc_mae_pred is {acc_mae_pred}')
    print(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_pred['many']['l1'],
                                                                             shot_dict_pred['median']['l1'], shot_dict_pred['low']['l1']) + "\n")
    print(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_gt['many']['l1'],
                                                                     shot_dict_gt['median']['l1'], shot_dict_gt['low']['l1']) + "\n")
    print('G-mean Gt {}, Many :  G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_gt, shot_dict_gt['many']['gmean'],
                                                                         shot_dict_gt['median']['gmean'], shot_dict_gt['low']['gmean'])+ "\n")                                                       
    print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_dict_pred['many']['gmean'],
                                                                         shot_dict_pred['median']['gmean'], shot_dict_pred['low']['gmean'])+ "\n")                                                       

    

    