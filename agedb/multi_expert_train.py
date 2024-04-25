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
import numpy  as np

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
parser.add_argument('--enable', action='store_false')




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
    test_dataset.enable_multi_crop(args.enable)
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False, train_labels = train_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, group_list, train_labels



def get_model(args):
    model = Encoder_regression_multi_expert(groups=3, name='resnet50')
    # load pretrained
    ''''
    if args.pretrained:
        ckpt = torch.load('last.pth')
        new_state_dict = OrderedDict()
        for k,v in ckpt['model'].items():
            key = k.replace('module.','')
            keys = key.replace('encoder.','')
            new_state_dict[keys]=v
        model.encoder.load_state_dict(new_state_dict)
    '''
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
    maj_shot, med_shot, min_shot = shot_count(train_labels)
    for e in tqdm(range(args.epoch)):
        maj_loss, med_loss, min_loss = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for idx, (x, y, g) in enumerate(train_loader):
            bsz = x.shape[0]
            index_list = find_regressors_index(y, maj_shot, med_shot, min_shot)
            loss = 0
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_output,  z = model(x)
            #
            for k in index_list.keys():
                g = index_list[k].to(device)
                bsz_g = g.shape[0]
                y_pred = torch.gather(y_output, dim=1, index=g.to(torch.int64)) 
                y_gt = torch.gather(y, dim=1, index=g.to(torch.int64)) 
                loss_mse = mse(y_pred, y_gt)
                loss += loss_mse
                metric = globals[f'{k}' + '_loss']
                metric.update(loss_mse.item(), bsz_g)
            #cls_loss.update(loss_ce.item(), bsz)
            loss.backward()
            opt.step()
        print(f' In epoch {e}, the majority mse loss is {maj_loss.avg}, the median mse loss is {med_loss.avg}, the minority mse loss is {min_loss.avg}')
    return model



def find_regressors_index(y, maj_shot, med_shot, min_shot ):
    return_index_list =  {}
    maj = torch.tensor(np.isin(y.numpy(),np.array(maj_shot)))
    maj_index = torch.nonzero(maj == True)
    if len(maj_index) != 0:
        return_index_list['maj'] = maj_index
    med = torch.tensor(np.isin(y.numpy(),np.array(med_shot)))
    med_index = torch.nonzero(med == True)
    if len(med_index) != 0:
        return_index_list['med'] = med_index
    min = torch.tensor(np.isin(y.numpy(),np.array(min_shot)))
    min_index = torch.nonzero(min == True)
    if len(min_index) != 0:
        return_index_list['min'] = min_index
    return return_index_list





def test_output(model, test_loader, train_labels, args):
    model.eval()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
    aggregation_weight.data.fill_(1/3) 
    opt = torch.optim.SGD(aggregation_weight.parameters(), lr= 0.025,
                                momentum=0.9, weight_decay=5e-4)
    for idx, (x,y,g) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        xx = torch.chunk(x, 2, dim=1)
        x1, x2 = xx[0].unsqueeze(1), xx[1].unsqueeze(1)
        y1, y2 = model(x1), model(x2)
        expert1_output1 = y1[:,0]
        expert2_output1 = y1[:,1]
        expert3_output1 = y1[:,2]
        expert1_output2 = y2[:,0]
        expert2_output2 = y2[:,1]
        expert3_output2 = y2[:,2]
        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight)
        aggregation_output0 = aggregation_softmax[0].cuda() * expert1_output1 + aggregation_softmax[1].cuda() * expert2_output1 + aggregation_softmax[2].cuda() * expert3_output1
        aggregation_output1 = aggregation_softmax[0].cuda() * expert1_output2 + aggregation_softmax[1].cuda() * expert2_output2 + aggregation_softmax[2].cuda() * expert3_output2
        cos_similarity = cos(aggregation_output0, aggregation_output1).mean()
        loss =  - cos_similarity
        opt.zero_grad()
        loss.backward()
        opt.step()
    aggregation_weight.eval()
    test_loader.enable_multi_crop(False)
    # mae
    test_mae_pred = AverageMeter()
    # gmean
    criterion_gmean = nn.L1Loss(reduction='none')
    pred, label = [], []
    for idx, (x,y,g) in enumerate(test_loader):
        with torch.no_grad():
            bsz = x.shape[0]
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            expert1 = y_pred[:,0]
            expert2 = y_pred[:,1]
            expert3 = y_pred[:,2]
            aggregation_output = aggregation_softmax[0].cuda() * expert1 + aggregation_softmax[1].cuda() * expert2 + aggregation_softmax[2].cuda() * expert3
            test_mae = F.l1_loss(aggregation_output, y)
            pred.extend(aggregation_output.cpu().numpy())
            label.extend(y.cpu().numpy())
            test_mae_pred.update(test_mae,bsz)
            loss_gmean = criterion_gmean(aggregation_output, y)
    shot_pred = shot_metrics(pred, label, train_labels)
    gmean_pred = gmean(np.hstack(loss_gmean), axis=None).astype(float)
    print(' Prediction Many: MAE {} Median: MAE {} Low: MAE {}'.format(shot_pred['many']['l1'],
                                                                    shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
    #
    print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                    shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n") 






if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader, group_list, train_labels = get_data_loader(args)
    model, optimizer = get_model(args)
    print(f' Start to train !')
    model = train_epoch(model, train_loader, val_loader, optimizer, args)
    test_output(model, test_loader, train_labels, args)


    
    
    
