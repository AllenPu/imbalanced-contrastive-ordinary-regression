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
from datasets.IMDBWIKI import *
from collections import OrderedDict
from loss import *
from loss_contra import *
from utils import *
import csv
#from OrdinalEntropy import *
import numpy  as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=0,
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
    df = pd.read_csv(os.path.join(args.data_dir, "imdb_wiki.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                   'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                          split='train', reweight=args.reweight, group_num=args.groups)
    #
    #group_list = train_dataset.get_group_list()
    #
    val_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = IMDBWIKI(data_dir=args.data_dir, df=df_test,
                         img_size=args.img_size, split='test', group_num=args.groups)
    #
    test_dataset1 = IMDBWIKI(data_dir=args.data_dir, df=df_test,
                         img_size=args.img_size, split='test', group_num=args.groups)
    test_dataset1.enable_multi_crop(args.enable)
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, test_loader1, train_labels



def get_model(args):
    model = Encoder_regression_multi_expert(name='resnet50')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer



def train_epoch(model, train_loader, train_labels, opt, args):
    model = torch.nn.DataParallel(model).cuda()
    #model = model.cuda()
    mse = nn.MSELoss()
    model.train()
    maj_shot, med_shot, min_shot = shot_count(train_labels)
    for e in tqdm(range(args.epoch)):
        for idx, (x, y, group, _) in enumerate(train_loader):
            bsz = x.shape[0]
            g = find_regressors_index(y, maj_shot, med_shot, min_shot)
            #print(f'y is {y} and g is {g}')
            x, y, g = x.cuda(non_blocking=True), y.cuda(non_blocking=True), g.cuda(non_blocking=True)
            opt.zero_grad()
            y_output = model(x)
            #
            y_pred = torch.gather(y_output, dim=1, index=g.to(torch.int64))
            #
            loss_mse = mse(y_pred, y)
            #
            loss_mse.backward()
            opt.step()
    return model



def find_regressors_index(y, maj_shot, med_shot, min_shot ):
    g_index = torch.Tensor(size=(y.shape[0],1))
    maj = torch.tensor(np.isin(y.numpy(),np.array(maj_shot)))
    maj_index = torch.nonzero(maj == True)[:,0]  
    if len(maj_index) != 0:
        g_index[maj_index] = 0
    med = torch.tensor(np.isin(y.numpy(),np.array(med_shot)))
    med_index = torch.nonzero(med == True)[:,0]
    if len(med_index) != 0:
        g_index[med_index] = 1
    min = torch.tensor(np.isin(y.numpy(),np.array(min_shot)))
    min_index = torch.nonzero(min == True)[:,0]
    if len(min_index) != 0:
        g_index[min_index] = 2
    return g_index





def test_output(model, test_loader1, test_loader, train_labels, args):
    model.eval()
    #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    mse = torch.nn.MSELoss()
    aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
    aggregation_weight.data.fill_(1/3)
    opt = torch.optim.SGD([aggregation_weight], lr= 0.025,momentum=0.9, weight_decay=5e-4, nesterov=True)
    for idx, (x,y,g) in enumerate(test_loader1):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        xx = torch.chunk(x, 2, dim=1)
        x1, x2 = xx[0].squeeze(1), xx[1].squeeze(1)
        y1, y2 = model(x1), model(x2)
        expert1_output1 = y1[:,0]
        expert2_output1 = y1[:,1]
        expert3_output1 = y1[:,2]
        expert1_output2 = y2[:,0]
        expert2_output2 = y2[:,1]
        expert3_output2 = y2[:,2]
        #print(f' aggregation_weight is {aggregation_weight}')
        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight)
        aggregation_output0 = aggregation_softmax[0].cuda() * expert1_output1 + aggregation_softmax[1].cuda() * expert2_output1 + aggregation_softmax[2].cuda() * expert3_output1
        aggregation_output1 = aggregation_softmax[0].cuda() * expert1_output2 + aggregation_softmax[1].cuda() * expert2_output2 + aggregation_softmax[2].cuda() * expert3_output2
        #cos_similarity = cos(aggregation_output0, aggregation_output1).mean()
        mse_similarity = mse(aggregation_output0, aggregation_output1).mean()
        loss =  - mse_similarity
        opt.zero_grad()
        loss.backward()
        opt.step()
    #
    aggregation_weight.requires_grad=False
    # mae
    test_mae_pred = AverageMeter()
    # gmean
    criterion_gmean = nn.L1Loss(reduction='none')
    pred, label, gmeans = [], [], []
    for idx, (x,y,g) in enumerate(test_loader):
        with torch.no_grad():
            bsz = x.shape[0]
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
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
            gmeans.extend(loss_gmean.cpu().numpy())
    shot_pred = shot_metric(pred, label, train_labels)
    gmean_pred = gmean(np.hstack(gmeans), axis=None).astype(float)
    print(' Prediction Many: All {}  MAE {} Median: MAE {} Low: MAE {}'.format(test_mae_pred.avg, shot_pred['many']['l1'],
                                                                    shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
    #
    print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                    shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n") 






if __name__ == '__main__':
    args = parser.parse_args()
    #cudnn.benchmark = True
    setup_seed(args.seed)
    train_loader, val_loader, test_loader, test_loader1, train_labels = get_data_loader(args)
    model, optimizer = get_model(args)
    print(f' Start to train !')
    model = train_epoch(model, train_loader, train_labels, optimizer, args)
    test_output(model, test_loader1, test_loader, train_labels, args)


    
    
    
