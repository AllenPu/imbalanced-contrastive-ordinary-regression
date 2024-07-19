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
import datetime
from loss import *
from elr import *
import matplotlib.pyplot as plt
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
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
parser.add_argument('--beta', type=float, default=0.7)
parser.add_argument('--lambdas', type=float, default=3)
parser.add_argument('--write_down', action='store_true', help=' write down the validation result to the csv file')
parser.add_argument('--we', type=int, default=10)



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
    group_list = train_dataset.get_three_shots_num_list()
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                         img_size=args.img_size, split='test', group_num=args.groups)
    #
    #test_dataset1 = AgeDB(data_dir=args.data_dir, df=df_test,
    #                     img_size=args.img_size, split='test', group_num=args.groups)
    #test_dataset1.enable_multi_crop(args.enable)
    #train_dataset.enable_elr_index_return(True)
    #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    #test_loader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False,
    #                         num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, group_list, train_labels



def get_model(args):
    model = Encoder_regression_uncertainty(name='resnet18', weight_norm=args.weight_norm, norm = args.norm)
    # load pretrained
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    return model, optimizer



# we is warm up epoch
def warm_up(model, train_loader, opt, we=10):
    model = model.cuda()
    model.train()
    for e in tqdm(range(we)):
        for idx, (x, y, g) in enumerate(train_loader):
            x, y, g = x.cuda(non_blocking=True), y.cuda(non_blocking=True), g.cuda(non_blocking=True)
            opt.zero_grad()
            pred, uncertain = model(x)
            loss = torch.mean(torch.pow(pred-y,2))
            loss.backward()
            opt.step()
    return model





def train_epoch_uncertain(model, train_loader, val_loader, train_labels, opt, args):
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    model.train()
    #
    for e in tqdm(range(args.epoch)):
        #####
        if e % 5 == 0 and e != 0:
            var_dict = {}
            var_list = []
            y_pred = []
            y_gt = []
            for idx, (x, y, _) in enumerate(val_loader):         
                #
                x = x.cuda(non_blocking=True)
                #
                pred, _ = model(x)
                #
                y_pred.extend(pred.data.cpu().numpy())
                y_gt.extend(y.data.numpy())
            y_pred, y_gt = torch.Tensor(np.hstack(y_pred)), np.hstack(y_gt)
            #
            aa = []
            for l in range(np.max(train_labels)+1):
                indexs = np.argwhere(y_gt==l).squeeze(-1)
                aa.append(l)
                if l not in y_gt or len(indexs) == 1:
                    variance = 0
                else:
                    index = torch.LongTensor(indexs)#.cuda()
                    variance = torch.var(y_pred.index_select(0, index)).item()
                var_dict[l] = variance  
                var_list.append(variance)  
                var_tensor = torch.Tensor(var_list)  
        else:
            var_tensor = torch.zeros(np.max(train_labels)+1)              
        ######
        for idx, (x, y, g) in enumerate(train_loader):
            bsz = x.shape[0]
            #
            #varianc_index = torch.LongTensor(y.squeeze(-1))
            #print(y.dtype)
            varianc = var_tensor.index_select(0, index= y.squeeze(-1).to(torch.int32))
            #
            varianc = varianc.unsqueeze(-1).cuda(non_blocking=True)
            #
            x, y, g = x.cuda(non_blocking=True), y.cuda(non_blocking=True), g.cuda(non_blocking=True)
            #
            pred, uncertain = model(x)
            #
            loss_mse = torch.mean(torch.pow(pred - y, 2))
            #
            opt.zero_grad()
            loss_mse.backward()
            opt.step()
            #
            if e % 5 == 0:
                pred, uncertain = model(x)
                #
                loss_mse = torch.pow(pred - y, 2).data
                #
                sigma = torch.log(varianc)
                #
                loss = torch.mean(torch.exp(-uncertain)*loss_mse + torch.abs(uncertain-sigma))
                #
                opt.zero_grad()
                loss.backward()
                opt.step()          
                #

    return model





def variance_calculation(model, train_loader):
    y_gt, y_pred, y_uncertain = [], [], []
    for idx, (x, y, g) in enumerate(train_loader):
        with torch.no_grad():
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred, uncertain = model(x)
            sigma = torch.sqrt(torch.exp(torch.abs(uncertain)))
            y_gt.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_uncertain.extend(sigma.cpu().numpy())
        #
    labels = np.unique(y_gt).tolist()
    #
    y_gt, y_pred, y_uncertain = np.hstack(y_gt).tolist(), np.hstack(y_pred).tolist(), np.hstack(y_uncertain).tolist()
    #
    y_pred = [math.ceil(i) if (i-int(i))> 0.5 else math.floor(i) for i in y_pred]
    #
    gt_list, uncertain_list, pred_list = count_down(labels, y_gt, y_pred, y_uncertain)
    #
    plt.plot(labels, gt_list, 'r--', uncertain_list, 'bs',  pred_list,  'g^' )
    plt.legend()
    #plt.show()
    plt.savefig('./var_scatter.png')
    plt.clf()
    gt_data, uncertain_data, pred_data = np.array(gt_list), np.array(uncertain_list), np.array(pred_list)
    gt_data, uncertain_data, pred_data = \
        gt_data.reshape(gt_data.shape[0], 1), uncertain_data.reshape(uncertain_data.shape[0], 1), pred_data.reshape(pred_data.shape[0], 1)
    datas = np.concatenate((gt_data, uncertain_data, pred_data),axis=1) 
    plt.hist(datas, bins=len(gt_list),edgecolor = 'w',color = ['c','r', 'b'],  label = ['gt','pred','pred_var'], stacked = False)
    ax = plt.gca() 
    plt.legend()
    #plt.show()
    plt.savefig('./var_hist.png')





def count_down(labels, y_gt, y_pred, y_uncertain):
    gt_list, pred_list, uncertain_list = [], [], []
    for i in labels:
        #
        indexes = [i for i,x in enumerate(y_pred) if x==i]
        gt_indexes = [i for i,x in enumerate(y_gt) if x==i]
        #
        gt = [y_gt[i] for i in gt_indexes]
        preds = [y_pred[i] for i in indexes]
        uncertains = [y_uncertain[i] for i in indexes]
        #
        gt_list.append(torch.sum(torch.Tensor(gt)).data)
        uncertain_list.append(torch.mean(torch.Tensor(uncertains)).data)
        pred_list.append(torch.sum(torch.Tensor(preds)).data)
    return gt_list, uncertain_list, pred_list
    
    






'''
def validates(model, val_loader, train_labels, maj_shot, med_shot, min_shot, e, store_name, write_down=False):
    pred, label, val_mae = [], [], AverageMeter()
    for idx, (x,y,_) in enumerate(val_loader):
        bsz = x.shape[0]
        with torch.no_grad():
            g = find_regressors_index(y, maj_shot, med_shot, min_shot)
            x, y, g = x.cuda(non_blocking=True), y.cuda(non_blocking=True), g.cuda(non_blocking=True)
            cls_pred, y_output = model(x)
            y_pred = torch.gather(y_output, dim=1, index=g.to(torch.int64))
            pred.extend(y_pred.cpu().numpy())
            label.extend(y.cpu().numpy())
            mae = F.l1_loss(y_pred, y)
            val_mae.update(mae, bsz)
    shot_pred = shot_metric(pred, label, train_labels)
    maj, med, low = shot_pred['many']['l1'], shot_pred['median']['l1'], shot_pred['low']['l1']
    print(f' In Epoch {e} total validation MAE is {val_mae.avg} MAE {maj} Median: MAE {med} Low: MAE {low}')
    _, _, _, min_to_med, min_to_maj, med_to_maj,med_to_min, maj_to_min,maj_to_med = shot_reg(label, pred, maj_shot, med_shot, min_shot)
    print(f'min_to_med {min_to_med}, min_to_maj {min_to_maj}, med_to_maj {med_to_maj}, med_to_min {med_to_min}, maj_to_min {maj_to_min}, maj_to_med {maj_to_med}')
    if write_down:
        with open(f'{store_name}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e, min_to_med, min_to_maj, med_to_maj,med_to_min, maj_to_min,maj_to_med])


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
'''




def test_output(model,  test_loader, train_labels, args):
    model.eval()
    maj_shot, med_shot, min_shot = shot_count(train_labels)
    #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #ce = torch.nn.CrossEntropyLoss()
    #
    test_mae_pred = AverageMeter()
    preds, label, gmeans = [], [], []
    criterion_gmean = nn.L1Loss(reduction='none')
    #
    for idx, (x,y,g) in enumerate(test_loader):
        with torch.no_grad():
            bsz = x.shape[0]
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred, uncertain = model(x)
            test_mae = F.l1_loss(pred, y)
            preds.extend(pred.cpu().numpy())
            label.extend(y.cpu().numpy())
            test_mae_pred.update(test_mae,bsz)
            #
            loss_gmean = criterion_gmean(pred, y)
            gmeans.extend(loss_gmean.cpu().numpy())
    store_name = 'bias_prediction_' + 'norm_' + str(args.norm) + '_weight_norm_' + str(args.weight_norm)
   # e = 0
    #
    #validates(model, test_loader, train_labels, maj_shot, med_shot, min_shot, e, store_name, write_down=False)
    shot_pred = shot_metric(preds, label, train_labels)
    gmean_pred = gmean(np.hstack(gmeans), axis=None).astype(float)
    #
    #variance_calculation(model, test_loader)
    #
    print(' Prediction All {}  Many: MAE {} Median: MAE {} Low: MAE {}'.format(test_mae_pred.avg, shot_pred['many']['l1'],
                                                                    shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
    #
    print(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                    shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n") 






if __name__ == '__main__':
    args = parser.parse_args()
    #
    today=datetime.date.today()
    #
    model_name =  'norm_' + str(args.norm) + '_weight_norm_' + str(args.weight_norm) + \
        '_epoch_' + str(args.epoch) + '_lr_' + str(args.lr) + '_' + str(today)
    #cudnn.benchmark = True
    setup_seed(args.seed)
    #
    train_loader, val_loader, test_loader, group_list, train_labels = get_data_loader(args)
    #
    model, optimizer= get_model(args)
    print(f' Start to warm up !')
    model = warm_up(model, train_loader, optimizer, args.we)
    #print('-----------------------------')
    #test_output(model, test_loader, test_loader, train_labels, args)
    print(f' Start to train !')
    model = train_epoch_uncertain(model, train_loader, val_loader, train_labels, optimizer, args)
    test_output(model, test_loader, train_labels, args)


    
    
    
