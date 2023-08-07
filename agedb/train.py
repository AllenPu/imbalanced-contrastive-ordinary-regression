import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.agedb import *
from utils import AverageMeter, accuracy, shot_metric, setup_seed, balanced_metrics, shot_metric_balanced
import torch
from loss import *
from network import *
import torch.optim as optim


import os
os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training/optimization related
parser.add_argument('--seed', default=3407)
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/home/ruizhipu/scratch/regression/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint',
                    help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='',
                    help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=[
                    'mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int,
                    default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
#
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--la', type=bool, default=False,
                    help='if use logit adj to train the imbalance')
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
parser.add_argument('--g_dis', action='store_true',
                    help='if dynamically adjust the tradeoff')
parser.add_argument('--gamma', type=float, default=5, help='tradeoff rate')
parser.add_argument('--reweight', type=str, default=None,
                    help='weight : inv or sqrt_inv')
#
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
#
parser.add_argument('--tau', default=1, type=float,
                    help=' tau for logit adjustment ')
parser.add_argument('--ranked_contra', action='store_true')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_data_loader(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                          split='train', reweight=args.reweight, group_num=args.groups)
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


def train_one_epoch(model, train_loader, ce_loss, mse_loss, opt, args):
    sigma, ranked_contra, contra_ratio, temp, g_dis, gamma = args.sigma, args.ranked_contra, args.contra_ratio, args.temp, args.g_dis, args.gamma
    model.train()
    ranges = int(100/args.groups)
    #
    for idx, (x, y, g) in enumerate(train_loader):
        #print('shape is', x.shape, y.shape, g.shape)
        #
        opt.zero_grad()
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        y_output, z = model(x)
        #
        y_chunk = torch.chunk(y_output, 2, dim=1)
        g_hat, y_pred = y_chunk[0], y_chunk[1]
        #
        g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
        #print('g_hat ', g_hat)
        #
        y_hat = torch.gather(y_pred, dim=1, index=g.to(torch.int64))
        #
        loss_mse = mse_loss(y_hat, y)
        #
        if g_dis:
            tol = 5/tolerance(g_index.cpu(), g.cpu(), ranges)
            sigma = gamma/tol
            loss_mse = sigma*loss_mse
        else:
            loss_mse = sigma*loss_mse
        #
        loss_ce = ce_loss(g_hat, g.squeeze().long())
        #
        if ranked_contra:
            loss_contra = contra_ratio * Ranked_Contrastive_Loss(z, g, temp=temp)
            loss = loss_ce + loss_contra + loss_mse
        else:
            loss = loss_ce + loss_mse
        #

        #
        loss.backward()
        opt.step()
    return model


def test(model, test_loader, train_labels, args):
    model.eval()
    #
    mse_gt = AverageMeter()
    mse_pred = AverageMeter()
    acc_g = AverageMeter()
    acc_mae_gt = AverageMeter()
    acc_mae_pred = AverageMeter()
    #
    pred_gt, pred, labels = [], [], []
    #
    for idx, (x, y, g) in enumerate(test_loader):
        bsz = x.shape[0]
        x, y, g = x.to(device), y.to(device), g.to(device)
        #
        labels.extend(y.data.cpu().numpy())
        # for cls, cls for g
        #
        with torch.no_grad():
            #
            y_output, _ = model(x)
            #
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_pred = y_chunk[0], y_chunk[1]
            #
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            #
            y_hat = torch.gather(y_pred, dim=1, index=g_index)
            y_pred_gt = torch.gather(y_pred, dim=1, index=g.to(torch.int64))
            #
            acc3 = accuracy(g_hat, g, topk=(1,))
            mae_y = torch.mean(torch.abs(y_hat - y))
            mae_y_gt = torch.mean(torch.abs(y_pred_gt - y))
            #
            pred.extend(y_hat.data.cpu().numpy())
            pred_gt.extend(y_pred_gt.data.cpu().numpy())
            #
        acc_g.update(acc3[0].item(), bsz)
        acc_mae_gt.update(mae_y_gt.item(), bsz)
        acc_mae_pred.update(mae_y.item(), bsz)
    shot_pred = shot_metric(pred, labels, train_labels)
    shot_pred_gt = shot_metric(pred_gt, labels, train_labels)

    return acc_g.avg, acc_mae_gt.avg, acc_mae_pred.avg, shot_pred, shot_pred_gt


def validate(model, val_loader, train_labels):
    model.eval()
    mae_pred = AverageMeter()
    preds, labels, preds_gt = [], [], []
    for idx, (x, y, g) in enumerate(val_loader):
        bsz = x.shape[0]
        x, y, g = x.to(device), y.to(device), g.to(device)
        with torch.no_grad():
            y_output,_ = model(x.to(torch.float32))
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_hat = y_chunk[0], y_chunk[1]
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            y_predicted = torch.gather(y_hat, dim=1, index=g_index)
            y_pred = torch.gather(y_hat, dim=1, index=g_index)
            mae = torch.mean(torch.abs(y_predicted - y))
            preds.extend(y_pred.data.cpu().numpy())
            labels.extend(y.data.cpu().numpy())
            preds_gt.extend(y_predicted.data.cpu().numpy())
        mae_pred.update(mae.item(), bsz)

    _, mean_L1_pred = balanced_metrics(np.hstack(preds), np.hstack(labels))
    _, mean_L1_gt = balanced_metrics(np.hstack(preds_gt), np.hstack(labels))
    shot_pred = shot_metric_balanced(preds, labels, train_labels)
    shot_pred_gt = shot_metric_balanced(preds_gt, labels, train_labels)
    return mae_pred.avg, mean_L1_pred, mean_L1_gt, shot_pred, shot_pred_gt


def write_log(store_name, results, shot_dict_pred, shot_dict_gt, args):
    with open(store_name, 'a+') as f:
        [ g_pred, mae_gt, mae_pred] = results
        f.write('---------------------------------------------------------------------')
        f.write(' tau is {} group is {} lr is {} model depth {} epoch {}'.format(
            args.tau, args.groups, args.lr, args.model_depth, args.epoch) + "\n")
        f.write(' acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format( g_pred, mae_gt, mae_pred)+"\n")
        #
        f.write(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_pred['many']['l1'],
                                                                             shot_dict_pred['median']['l1'], shot_dict_pred['low']['l1']) + "\n")
        #
        f.write(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_gt['many']['l1'],
                                                                     shot_dict_gt['median']['l1'], shot_dict_gt['low']['l1']) + "\n")
        #
        #f.write(' CLS Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_cls['many']['cls'], \
        #                                                                       shot_dict_cls['median']['cls'], shot_dict_cls['low']['cls'])+ "\n" )
        #
        f.write('---------------------------------------------------------------------')
        f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    store_names = 'la_' + str(args.la) + '_tau_' + str(args.tau) + \
        '_lr_' + str(args.lr) + '_g_' + str(args.groups) + '_model_' + str(args.model_depth) + \
        '_epoch_' + str(args.epoch) + '_group_dis_' + str(args.g_dis) + '_sigma_' + str(args.sigma) + \
        '_gamma_' + str(args.gamma) + str(args.reweight)
    ####
    print(" store name is ", store_names)
    #
    store_name = store_names + '.txt'
    #
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_data_loader(
        args)
    #
    loss_mse = nn.MSELoss()
    loss_ce = LAloss(cls_num_list, tau=args.tau).to(device)
    #
    model = ResNet_regression(args).to(device)
    model_val = ResNet_regression(args).to(device)
    #
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #
    for e in tqdm(range(args.epoch)):
        model = train_one_epoch(model, train_loader,
                                loss_ce, loss_mse, opt, args)
        if e % 20 == 0 or e == (args.epoch - 1):
            reg_mae,  mean_L1_pred,  mean_L1_gt, shot_dict_val_pred, shot_dict_val_pred_gt = validate(
                model, val_loader, train_labels)
            #
            if best_bMAE > mean_L1_pred and e > 40:
                best_bMAE = mean_L1_pred
                torch.save(model.state_dict(),
                           './models/model_{}.pth'.format(store_names))
            with open(store_name, 'a+') as f:
                f.write(
                    '---------------------------------------------------------------------\n')
                f.write(' In epoch {} gt regression mae is {} best bMAE is {}'.format(
                    e, reg_mae, best_bMAE) + '\n')
                f.write(' Val bMAE is pred {}, bMAE is gt {}'.format(
                    mean_L1_pred,  mean_L1_gt) + '\n')
                f.write(' Val Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_val_pred['many']['l1'],
                                                                                         shot_dict_val_pred['median']['l1'], shot_dict_val_pred['low']['l1']) + "\n")
                f.write(' Val Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_dict_val_pred_gt['many']['l1'],
                                                                                 shot_dict_val_pred_gt['median']['l1'], shot_dict_val_pred_gt['low']['l1']) + "\n")
                f.write(
                    '---------------------------------------------------------------------\n')
                f.close()
    acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, shot_pred, shot_pred_gt = test(
        model, test_loader, train_labels, args)
    results = [acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg]
    write_log(store_name, results, shot_pred, shot_pred_gt, args)
    #
    model_val.load_state_dict(torch.load(
        './models/model_{}.pth'.format(store_names)))
    acc_g_avg_val, acc_mae_gt_avg_val, acc_mae_pred_avg_val, shot_pred_val, shot_pred_gt_val = \
                                                                                test(model_val, test_loader, train_labels, args)
    results_test = [acc_g_avg_val, acc_mae_gt_avg_val, acc_mae_pred_avg_val]
    write_log(store_name, results_test, shot_pred_val, shot_pred_gt_val, args)
