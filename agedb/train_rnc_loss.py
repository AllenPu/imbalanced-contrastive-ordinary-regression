import pandas as pd
import os
import torch
import time
import argparse
from tqdm import tqdm
import pandas as pd
from network import *
from model import *
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.agedb import *
from collections import OrderedDict
from loss import *
from loss_contra import *
from utils import *
from train import test, write_log
from util_devlove import shot_metrics, train_regressor, validate
from draw_tsne import draw_tsne
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--epoch', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--reweight', type=str, default=None,
                    help='weight : inv or sqrt_inv')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')
parser.add_argument('--output_file', type=str,
                    default='result_rnc', help='store')
parser.add_argument('--scale', type=float, default=1, help='scale of the sharpness in soft label')
parser.add_argument('--soft_label', action='store_true')
parser.add_argument('--ce', action='store_true',  help='if use the cross_entropy /la or not')
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--la', action='store_true')
parser.add_argument('--mse', action='store_true')
parser.add_argument('--single_output', action='store_true')



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



def get_model(args):
    if args.single_output:
        model = Encoder_regression_single(name='resnet18')
    else:
        model = Encoder_regression(groups=args.groups, name='resnet18')
    # load pretrained
    ckpt = torch.load('last.pth')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys]=v
    model.encoder.load_state_dict(new_state_dict)
    # freeze the pretrained part
    #for (name, param) in model.encoder.named_parameters():
    #    param.requires_grad = False
    #
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer



def train_epoch(model, train_loader, val_loader, opt, args):
    model = model.to(device)
    mse = nn.MSELoss()
    for e in tqdm(range(args.epoch)):

        cls_loss = AverageMeter()
        mse_loss = AverageMeter()
        model.train()
        for idx, (x, y, g) in enumerate(train_loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
            opt.zero_grad()
            y_output,  z = model(x)
            #
            y_ =  torch.chunk(y_output,2,dim=-1)
            g_hat, y_hat = y_[0], y_[1]
            y_pred = torch.gather(y_hat, dim=1, index=g.to(torch.int64)) 
            g_pred = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            if args.soft_label:
                prefix = 'soft_label'
                g_soft_label = soft_labeling(g, args).to(device)
                loss_ce = SoftCrossEntropy(g_hat, g_soft_label)
                #print(f' soft label loss is {loss_ce.item()}')
            elif args.ce:
                prefix = 'ce'
                loss_ce = F.cross_entropy(g_hat, g.squeeze().long(), reduction='mean')
            elif args.la :
                loss_la = LAloss(group_list)
                loss_ce = loss_la(g_hat, g.squeeze().long())
                prefix = 'la'
            elif args.mse:
                loss_ce = mse(g_pred, g)
                loss_ce = 0
                #print(f' ce loss is {loss_ce.item()}')
            #if torch.isnan(loss_ce):
            #    print(f' g_hat is {g_hat[:10]} g is {g[:10]} z is {z[:10]}')
            #    assert 1==0
            loss_mse = mse(y_pred, y)
            loss = loss_mse + loss_ce
            cls_loss.update(loss_ce.item(), bsz)
            mse_loss.update(loss_mse.item(), bsz)
            loss.backward()
            opt.step()
        #
       
        model.eval()
        val_cls_loss = AverageMeter()
        val_mse_loss = AverageMeter()
        val_mse_loss_gt = AverageMeter()
        for idx, (x, y, g) in enumerate(train_loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
            with torch.no_grad():
                y_output,_ = model(x.to(torch.float32))
                y_chunk = torch.chunk(y_output, 2, dim=1)
                g_hat, y_hat = y_chunk[0], y_chunk[1]
                g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
                y_pred = torch.gather(y_hat, dim=1, index=g_index.to(torch.int64))
                y_gt = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
                # 
                val_cls = F.cross_entropy(g_hat, g.squeeze().long())
                val_mse = F.mse_loss(y_pred, y)
                val_mse_gt = F.mse_loss(y_gt, y)
                val_cls_loss.update(val_cls.item(), bsz)
                val_mse_loss.update(val_mse.item(), bsz)
                val_mse_loss_gt.update(val_mse_gt.item(), bsz)
        #print(f' At Epoch {e}, val cls loss is {val_cls_loss.avg} val mse loss is {val_mse_loss.avg}')
        with open(f'./{prefix}_loss_2.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e,cls_loss.avg,mse_loss.avg, val_cls_loss.avg, val_mse_loss.avg, val_mse_loss_gt.avg])
        print(f' At Epoch {e}, cls loss is {cls_loss.avg}, mse loss is {mse_loss.avg} val cls loss is {val_cls_loss.avg} val mse loss is {val_mse_loss.avg}  val mse loss gt is {val_mse_loss_gt.avg}')
    return model






def train_epoch_single(model, train_loader, val_loader, train_labels,  opt, args):
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()   
    maj, med, mino = shot_count(train_labels)
    for e in tqdm(range(args.epoch)):
        mse_loss = AverageMeter()
        val_mse_loss = AverageMeter()
        # how many labels are wrongly predicted
        y_dis, labels, preds = {}, [], []
        for idx, (x, y, g) in enumerate(train_loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
            opt.zero_grad()
            y_output,  z = model(x)
            loss_mse = mse(y_output, y)
            loss = loss_mse 
            mse_loss.update(loss_mse.item(), bsz)
            loss.backward()
            opt.step()
            preds.extend(y_output.data.cpu().numpy())
            labels.extend(y.data.cpu().numpy())
            #
            # added for how many labels are wrongly predicted in training
        '''
            dis = torch.floor(torch.abs(y - y_output))
            for items in range(10):
                y_dis[items] = y_dis.get(items,0) + (dis == items ).sum(dim=0).item()
                #print(f'dis is {dis} result is {(dis == items ).sum(dim=0)}')
            
        val_mse_loss = AverageMeter()
        for idx, (x, y, g) in enumerate(val_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            with torch.no_grad():
                y_output,_ = model(x.to(torch.float32))
                val_mse = F.mse_loss(y_output, y)
                val_mse_loss.update(val_mse.item(), bsz)
        with open('./single_loss.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e,mse_loss.avg, val_mse_loss.avg])
        print(f' At Epoch {e} single mse loss is {mse_loss.avg} val loss is {val_mse_loss.avg}')
        '''
        pred_maj, pred_med, pred_min, pred_min_to_med, pred_min_to_maj, pred_med_to_maj, pred_min_to_med = shot_reg(labels, preds, maj, med, mino)
        with open('./prediction_bias2.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            write_list = []
            #for items in range(10):
            #    write_list.append(y_dis[items])
            writer.writerow([e, pred_maj, pred_med, pred_min, pred_min_to_med, pred_min_to_maj, pred_med_to_maj, pred_min_to_med ])
        #'''
    return model




def shot_count(train_labels, many_shot_thr=100, low_shot_thr=20):
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count = []
    #
    maj_class, med_class, min_class = [], [], []
    #
    for l in np.unique(train_labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
    #
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            maj_class.append(i)
        elif train_class_count[i] < low_shot_thr:
            min_class.append(i)
        else:
            med_class.append(i) 
    #
    return maj_class, med_class, min_class


def shot_reg(label, pred, maj, med, min):
    # how many preditions in this shots
    pred_dict = {'maj':0, 'med':0, 'min':0}
    # how many preditions from min to med, min to maj, med to maj, min to med
    pred_label_dict = {'min to med':0, 'min to maj':0, 'med to maj':0, 'min to med':0}
    labels, preds = np.stack(label), np.floor(np.hstack(pred))
    #dis = np.floor(np.abs(labels - preds)).tolist()
    bsz = labels.shape[0]
    for i in range(bsz):
        k_pred = check_shot(preds[i],maj, med, min)
        k_label = check_shot(labels[i],maj, med, min)
        if k_pred in pred_dict.keys():
            pred_dict[k_pred] = pred_dict[k_pred] + 1
        pred_shift = check_pred_shift(k_pred, k_label)
        if pred_shift in pred_label_dict.keys():
            pred_label_dict[pred_shift] = pred_label_dict[pred_shift] + 1
    return pred_dict['maj'], pred_dict['med'], pred_dict['min'], pred_label_dict['min to med'], pred_label_dict['min to maj'], pred_label_dict['med to maj'], pred_label_dict['min to med']


def check_shot(e, maj, med, min):
    if e in maj:
        return 'maj'
    elif e in med:
        return 'med'
    else:
        return 'min'
    
# check reditions from min to med, min to maj, med to maj
def check_pred_shift(k_pred, k_label):
    if k_pred is 'med' and k_label is 'min':
        return 'min to med'
    elif k_pred is 'maj' and k_label is 'med':
        return 'med to maj'
    elif k_pred is 'maj' and k_label is 'min':
        return 'min to maj'
    elif k_pred is 'med' and k_label is 'min':
        return 'min to med'
    else:
        return 'others'

    



if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader, group_list, train_labels = get_data_loader(args)
    model, optimizer = get_model(args)
    store_name = 'groups_' + str(args.groups) + '_lr_' + str(args.lr) + '_epoch_' + str(args.epoch)
    if args.soft_label:
        prefix = '_soft_label'
    elif args.la:
        prefix = '_la'
    elif args.ce:
        prefix = '_ce'
    else:
        print(f'no classification criterion specified !!!')
        prefix = 'original_'
    store_name = store_name + prefix
    #encoder, regressor = train_regressor(train_loader, model.encoder, model.regressor, optimizer, args)
    #validate(val_loader, encoder, regressor, train_labels=train_labels)
    print(f' Start to train !')
    if args.single_output:
        model = train_epoch_single(model, train_loader,val_loader, train_labels, optimizer, args)
    else:
        model = train_epoch(model, train_loader, val_loader, optimizer, args)


    
    
    
