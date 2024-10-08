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




def tolerance(g_pred, g, ranges):
    # g_pred is the prediction tensor
    # g is the ground truth tensor
    # range is the fixed group range
    g = np.array(g)
    g_pred = np.array(g_pred)
    groups = {}
    #
    tolerance = 0
    #
    for l in np.unique(g):
        groups[l] = len(g[g == l])
    for l in groups.keys():
        index = np.where(g == l)[0]
        g_current_pred = g_pred[index]
        g_current_gt = g[index]
        var = np.sum(np.abs(g_current_pred - g_current_gt))
        bias_var = var/groups[l]
        tolerance += bias_var
    tolerance = tolerance/ranges
    #
    return tolerance


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
    #sigma, la, g_dis, gamma, ranked_contra, contra_ratio, soft_label, ce = \
    #    args.sigma, args.la, args.g_dis, args.gamma, args.ranked_contra, args.contra_ratio, args.soft_label, args.ce
    ranges = int(100/args.groups)
    #
    repeats = 1
    #
    model.train()
    #
    mse_y = 0
    ce_g = 0
    #
    tol = 0
    tole = []
    #
    if e == 0:
        if args.ranked_contra :
            group_loss = RnCLoss(temperature=args.temp).to(device)
            print(f' Group wise Contrastive loss initiated ')
        elif args.aug:
            sample_loss = RnCLoss_pairwise(temperature=args.temp).to(device)
            print(f' Pair wise Contrastive loss initiated ')            
        elif args.la:
            print(f' Logit adjustment initiated ')
            la_loss = LAloss(cls_num_list, tau=args.tau).to(device)
        elif args.ce:
            print(f' Cross Entropy initiated ')
            ce_loss = nn.CrossEntropyLoss()
    #
    for idx, (x, y, g, w) in enumerate(train_loader):
        bs = y.shape[0]
        opt.zero_grad()
        # x shape : (batch,channel, H, W)
        # y shape : (batch, 1)
        # g hsape : (batch, 1)
        #print(' g is ', g)
        y, g, w =  y.to(device), g.to(device), w.to(device)
        #
        if args.aug:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            #x = torch.cat([x[0], x[1]], dim=0)
            x = x.to(device)
            repeats = 2
        else:
            x = x.to(device)
        #
        y_output, z = model(x)
        #split into two parts : first is the group, second is the prediction
        y_chunk = torch.chunk(y_output, 2, dim=1)
        #
        g_hat, y_hat = y_chunk[0], y_chunk[1]
        #
        #extract y out
        y_predicted = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
        #
        loss = 0
        loss_list = []
        #
        # rewrite mse loss if reweight
        #
        if args.reweight is not None:
            mse_y = (y_predicted - y) ** 2
            w = w.to(device)
            mse_y *= w.expand_as(mse_y)
            mse_y = torch.mean(mse_y)
        else:
            mse_y = mse_loss(y_predicted, y)
        #
        loss_list.append(args.sigma*mse_y)
        #
        if args.la:
            la_g = la_loss(g_hat, g.squeeze().long())
            loss_list.append(la_g)
        #
        if args.ce:
            ce_g = ce_loss(g_hat, g.squeeze().long())
            #
            if args.smooth:
                ce_g = ce_g * w.squeeze(-1)
                ce_g = torch.mean(ce_g)
            loss_list.append(ce_g)             
        # group wise contrastive loss
        if args.ranked_contra :
            ranked_contrastive_loss = args.contra_ratio * group_loss(z, g)
            loss_list.append(ranked_contrastive_loss)  
        # sample wise contrastive loss
        if args.aug:
            f1, f2 = torch.split(z, [bs, bs], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            pairwise_contrastive_loss = args.contra_ratio * sample_loss(features, g)
            loss_list.append(pairwise_contrastive_loss)  
        #
        if args.g_dis:
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            tol = tolerance(g_index.cpu(), g.cpu(), ranges)
            sigma = args.gamma/tol
            tole.append(tol)
        #
        if args.soft_label:
            g_soft = soft_labeling(g, args).to(device)
            g_soft_label = g_soft.repeat(repeats, 1)
            loss_soft_g = SoftCrossEntropy(g_hat, g_soft_label)
            loss_list.append(loss_soft_g)
            #print(' soft g is ', g)
            #print(f'soft label loss is {loss_soft_g.item()}')
        #
        if args.diversity != 0:
            diversity_loss = args.diversity * feature_diversity(z, g, args)
            loss_list.append(diversity_loss)
        #
        #print(f'loss list is {loss_list}')
        #loss = mse_y + sigma*ce_g
        for i in loss_list:
            loss += i
        loss.backward()
        opt.step()
        #
        #if idx%50 == 0:
        #    tol= tolerance(g_index.cpu() , g.cpu(), ranges)
        #   print(" tolerance ", tol)
        #
    if not args.g_dis:
        tole = [0]
    
    tol_avg = int(np.mean(tole))
    #if tol_avg == 1:
    #    print(" current epoch is ", e)
    #    print(tole)
    return model, tol_avg


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
    group_and_pred = {'pred': [], 'gt':[]}
    #
    pred_gt, pred, labels, groups = [], [], [], []
    # CHECK THE PREDICTION ACC
    pred_g_gt, pred_g = [], []
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
            # initi for tsne
            #tsne_x_gt = torch.Tensor(0)
            #
            y_output, z = model(inputs)
            #
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_hat = y_chunk[0], y_chunk[1]
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
            groups.extend(group.data.cpu().numpy())
            #
            #group_and_pred['pred'].extend(g_index.data.cpu().numpy())
            #group_and_pred['gt'].extend(group.data.cpu().numpy())
            #
            mse_y_gt = mse(y_gt, targets)
            mse_y_pred = mse(y_pred, targets)
            #mse_mean_1 = mse(y_predicted_mean, targets)
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
    #
    #
    return mse_gt.avg,  mse_pred.avg, acc_g.avg, acc_mae_gt.avg, acc_mae_pred.avg,\
                                    shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred,  group_and_pred


def validate(model, val_loader, train_labels, e):
    model.eval()
    g_cls_acc = AverageMeter()
    y_gt_mae = AverageMeter()
    preds, labels, preds_gt = [], [], []
    for idx, (inputs, targets, group, _) in enumerate(val_loader):
        inputs, targets, group = inputs.to(
            device), targets.to(device), group.to(device)
        bsz = inputs.shape[0]
        with torch.no_grad():
            y_output, z = model(inputs.to(torch.float32))
            ##
            #
            y_chunk = torch.chunk(y_output, 2, dim=1)
            #
            g_hat, y_hat = y_chunk[0], y_chunk[1]
            #
            y_predicted = torch.gather(
                y_hat, dim=1, index=group.to(torch.int64))
            #
            #
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            y_pred = torch.gather(y_hat, dim=1, index=g_index)
            #
            acc = accuracy(g_hat, group, topk=(1,))
            mae = torch.mean(torch.abs(y_predicted - targets))
            #
            preds.extend(y_pred.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            preds_gt.extend(y_predicted.data.cpu().numpy())
        #
        g_cls_acc.update(acc[0].item(), bsz)
        y_gt_mae.update(mae.item(), bsz)
        #
    #torch.save(preds, './val/pred_{}.pt'.format(e))
    #torch.save(labels, './val/labels_{}.pt'.format(e))
    _, mean_L1_pred = balanced_metrics(np.hstack(preds), np.hstack(labels))
    _, mean_L1_gt = balanced_metrics(np.hstack(preds_gt), np.hstack(labels))
    #
    shot_dict_pred = shot_metric_balanced(preds, labels, train_labels)
    shot_dict_pred_gt = shot_metric_balanced(preds_gt, labels, train_labels)
    #
    return g_cls_acc.avg, y_gt_mae.avg, mean_L1_pred,  mean_L1_gt, shot_dict_pred, shot_dict_pred_gt


def write_test_loggs(store_name, results, shot_dict_pred, shot_dict_gt, shot_dict_cls, args, current_task_name=None, mode = None):
    with open(store_name, 'a+') as f:
        [acc_gt, acc_pred, g_pred, mae_gt, mae_pred, gmean_gt, gmean_pred] = results
        f.write('---------------------------------------------------------------------\n')
        if current_task_name is not None and mode is not None:
            f.write('  current task name is {}'.format(current_task_name) + "\n")
            f.write(' current mode is {}'.format(mode) + "\n")
        f.write(f' time is {time.asctime()}')
        f.write(' tau is {} group is {} lr is {} model depth {} epoch {} time {}'.format(
            args.tau, args.groups, args.lr, args.model_depth, args.epoch, time.asctime()) + "\n")
        f.write(' mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred)+"\n")
        #
        f.write(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_pred['many']['l1'],
                                                                             shot_dict_pred['median']['l1'], shot_dict_pred['low']['l1']) + "\n")
        #
        f.write(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_gt['many']['l1'],
                                                                     shot_dict_gt['median']['l1'], shot_dict_gt['low']['l1']) + "\n")
        #
        f.write(' CLS Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_cls['many']['cls'],
                                                                         shot_dict_cls['median']['cls'], shot_dict_cls['low']['cls']) + "\n")
        #
        f.write(' G-mean Gt {}, Many :  G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_gt, shot_dict_gt['many']['gmean'],
                                                                         shot_dict_gt['median']['gmean'], shot_dict_gt['low']['gmean'])+ "\n")                                                       
        #
        f.write(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_dict_pred['many']['gmean'],
                                                                         shot_dict_pred['median']['gmean'], shot_dict_pred['low']['gmean'])+ "\n")                                                       
        #
        f.write('---------------------------------------------------------------------\n')
        f.close()


def write_val_log(store_name, cls_acc, reg_mae,  mean_L1_pred,  mean_L1_gt, shot_dict_val_pred, shot_dict_val_pred_gt, tol):
    with open(store_name, 'a+') as f:
        f.write('---------------------------------------------------------------------\n')
        f.write(' In epoch {} cls acc is {} regression mae is {} best bMAE is {} tol is {}'.format(
                    e, cls_acc, reg_mae, best_bMAE, tol) + '\n')
        f.write(' Val bMAE is pred {}, bMAE is gt {}'.format(
                    mean_L1_pred,  mean_L1_gt) + '\n')
        f.write(' Val Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_val_pred['many']['l1'],
                    shot_dict_val_pred['median']['l1'], shot_dict_val_pred['low']['l1']) + "\n")
        f.write(' Val Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_val_pred_gt['many']['l1'],
                    shot_dict_val_pred_gt['median']['l1'], shot_dict_val_pred_gt['low']['l1']) + "\n")
        f.write('---------------------------------------------------------------------\n')
        f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    #
    #total_result = 'total_result_model_'+str(args.model_depth)+'.txt'
    #
    store_names = 'ce_' + str(args.ce) + '_la_' + str(args.la) + '_tau_' + str(args.tau) + '_lds_' + str(args.lds) + '_group_mode_' + str(args.group_mode) +\
        '_lr_' + str(args.lr) + '_g_' + str(args.groups) + '_model_' + str(args.model_depth) + \
        '_epoch_' + str(args.epoch) + '_bs_' + str(args.batch_size) + '_sigma_' + str(args.sigma) + \
        '_gamma_' + str(args.gamma) + '_contras_' + str(args.ranked_contra) + '_temp_' + str(args.temp) + \
        '_scale_' + str(args.scale) + '_feature_diversity_' + str(args.diversity) + '_smooth_data_' + str(args.smooth) + '_aug_' + str(args.aug)
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
    # focal loss
    #if args.fl:
    #    loss_ce = FocalLoss(gamma=0.75)
    #
    #print(" tau is {} group is {} lr is {} model depth {}".format(
    #    args.tau, args.groups, args.lr, args.model_depth))
    #
    best_bMAE = 100
    tole = []
    #
    #print(" raw model for group classification trained at epoch {}".format(e))
    for e in tqdm(range(args.epoch)):
        #adjust_learning_rate(opt, e, args)
        model, tol = train_one_epoch(
            model, train_loader, loss_reg, opt, args, e)
        tole.append(tol)
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
    #
    # val best model
    #
    results_val = [acc_gt, acc_pred, g_pred, mae_gt, mae_pred, gmean_gt, gmean_pred]
    write_test_loggs('./output/' + store_name, results_val, shot_dict_pred,
                shot_dict_gt, shot_dict_cls, args)
    #
    if args.ranked_contra:
        filename = args.output_file + '_contra.txt'
        write_test_loggs(filename, results_val, shot_dict_pred,
                shot_dict_gt, shot_dict_cls, args, current_task_name=store_names, mode = 'val')
    else:
        filename = args.output_file + 'no_contra.txt'
        write_test_loggs(filename, results_val, shot_dict_pred,
                shot_dict_gt, shot_dict_cls, args, current_task_name=store_names, mode = 'val')
    #
    # test train model
    #
    acc_gt, acc_pred, g_pred, mae_gt, mae_pred, shot_dict_pred, shot_dict_gt, shot_dict_cls, gmean_gt, gmean_pred, group_and_pred = \
        test_step(model, test_loader, train_labels, args)
    print(' Test model mse of gt is {}, mse of pred is {}, acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_gt, acc_pred, g_pred, mae_gt, mae_pred))
    results_test = [acc_gt, acc_pred, g_pred, mae_gt, mae_pred, gmean_gt, gmean_pred ]
    write_test_loggs('./output/'+store_name, results_test, shot_dict_pred,
                shot_dict_gt, shot_dict_cls, args)
    #
    # check if used  eq groups
    with open('test_gt.pkl', 'wb') as f:
        pickle.dump(group_and_pred, f)
    #
    if args.ranked_contra:
        filename = args.output_file + '_contra.txt'
        write_test_loggs(filename, results_test, shot_dict_pred,
                     shot_dict_gt, shot_dict_cls, args, current_task_name=store_names, mode='test')
    else:
        filename = args.output_file + 'no_contra.txt'
        write_test_loggs(filename, results_val, shot_dict_pred,
                         shot_dict_gt, shot_dict_cls, args, current_task_name=store_names, mode='test')
    

   
