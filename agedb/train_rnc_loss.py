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
from OrdinalEntropy import *

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
parser.add_argument('--oe', action='store_true', help='ordinal entropy')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--weight_norm', action='store_true')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--frob', action='store_true')
parser.add_argument('--frob_lambda', default=1.0, type=float)


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
        model = Encoder_regression_single(name='resnet18', norm=args.norm, weight_norm=args.weight_norm)
    else:
        model = Encoder_regression(groups=args.groups, name='resnet18')
    # load pretrained
    if args.pretrained:
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
    #maj, med, mino = shot_count(train_labels)
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
            if args.oe:
                loss += ordinalentropy(z, y)
            if args.frob:
                frobenius_norm = torch.mean(torch.norm(z, p='fro', dim=-1))
                loss += args.frob_lambda * frobenius_norm
            mse_loss.update(loss_mse.item(), bsz)
            loss.backward()
            opt.step()
            #
            #preds.extend(y_output.data.cpu().numpy())
            #labels.extend(y.data.cpu().numpy())
            #
            # added for how many labels are wrongly predicted in training
        '''
        # validation
        labels_val, preds_val = [], []
        val_mse_loss = AverageMeter()
        for idx, (x, y, g) in enumerate(val_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            with torch.no_grad():
                y_output,_ = model(x.to(torch.float32))
                val_mse = F.mse_loss(y_output, y)
                val_mse_loss.update(val_mse.item(), bsz)
                preds_val.extend(y_output.data.cpu().numpy())
                labels_val.extend(y.data.cpu().numpy())
        with open('./single_loss.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e,mse_loss.avg, val_mse_loss.avg])
        print(f' At Epoch {e} single mse loss is {mse_loss.avg} val loss is {val_mse_loss.avg}')
        # claculate the bias between different shots
        pred_maj, pred_med, pred_min, pred_min_to_med, pred_min_to_maj, pred_med_to_maj, pred_med_to_min, pred_maj_to_min, pred_maj_to_med = \
            shot_reg(labels_val, preds_val, maj, med, mino)
        with open('./prediction_bias3_val.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            write_list = []
            #for items in range(10):
            #    write_list.append(y_dis[items])
            writer.writerow([e, pred_maj, pred_med, pred_min, pred_min_to_med, pred_min_to_maj, pred_med_to_maj, pred_med_to_min, pred_maj_to_min, pred_maj_to_med])
        '''
    #test
    
    return model


# test for single output network
def test_single(model, test_loader, train_labels):
    model.eval()
    #
    test_mae_pred = AverageMeter()
    pred, label = [], []
    # variables frob norm for each shot average
    majs, meds, mino = shot_count(train_labels)
    maj_shot_nuc, med_shot_nuc, min_shot_nuc = AverageMeter(), AverageMeter(), AverageMeter()
    maj_shot, med_shot, min_shot  = AverageMeter(), AverageMeter(), AverageMeter()
    #maj_shot, med_shot, min_shot = 0, 0, 0
    with torch.no_grad():
        for idx, (x, y, g) in enumerate(test_loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
            y_output, feat = model(x.to(torch.float32))
            test_mae = F.l1_loss(y_output, y)
            pred.extend(y_output.cpu().numpy())
            label.extend(y.cpu().numpy())
            test_mae_pred.update(test_mae,bsz)
            # calcualte frib norm for each shot average
            maj_shot, med_shot, min_shot,  maj_shot_nuc, med_shot_nuc, min_shot_nuc = cal_frob_norm(y, feat, majs, meds, mino, maj_shot, med_shot, min_shot, maj_shot_nuc, med_shot_nuc, min_shot_nuc)
        pred_shot = shot_metric(pred, label, train_labels)
    many , med, low = pred_shot['many']['l1'], pred_shot['median']['l1'], pred_shot['low']['l1']
    print(f' the mae of prediction is {test_mae_pred.avg}, the many shot is {many} median is {med} minority is {low}')
    #print(f' the norm of maj is {maj_shot/len(majs)}, the norm of med is {med_shot/len(meds)}, the norm of low is {min_shot/len(mino)}')
    print(f' the norm of maj is {maj_shot.avg}, the norm of med is {med_shot.avg}, the norm of low is {min_shot.avg}')
    print(f' the singular of maj is {maj_shot_nuc.avg}, the norm of med is {med_shot_nuc.avg}, the norm of low is {min_shot_nuc.avg}')



def test_output(model, test_loader, train_labels, args):
    acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, shot_pred, shot_pred_gt, gmean_gt, gmean_pred, = test(
        model, test_loader, train_labels, args)
    results = [acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, gmean_gt, gmean_pred]
    #write_log('./output/'+store_name, results, shot_pred, shot_pred_gt, args)
    #test_group_acc(model, train_loader, prefix)
    print(' acc of the group assinment is {}, \
            mae of gt is {}, mae of pred is {}'.format(acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg)+"\n")
        #
    print(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_pred['many']['l1'],
                                                                    shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
        #
    print(' Gt Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_pred_gt['many']['l1'],
                                                                    shot_pred_gt['median']['l1'], shot_pred_gt['low']['l1']) + "\n")
        #
    print(' G-mean Gt {}, Many :  G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_gt, shot_pred_gt['many']['gmean'],
                                                                    shot_pred_gt['median']['gmean'], shot_pred_gt['low']['gmean'])+ "\n")                                                       
        #
    print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                    shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n") 






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
        test_single(model, test_loader, train_labels)
    else:
        model = train_epoch(model, train_loader, val_loader, optimizer, args)
        test_output(model, test_loader, train_labels, args)


    
    
    
