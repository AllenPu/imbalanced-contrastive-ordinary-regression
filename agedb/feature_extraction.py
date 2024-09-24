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
#from utils import soft_labeling, SoftCrossEntropy
from util_devlove import shot_metrics, train_regressor, validate
from draw_tsne import draw_tsne

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
# margin of the groups , .e.g. for a 10 groups label, step = 2 [...,6,8,10,8,6,...]
parser.add_argument('--step', type=float, default=1)
parser.add_argument('--la', action='store_true')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--best', action='store_true')
#
parser.add_argument('--asymm', action='store_true', help='if use the asymmetric soft label')



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
    model = Encoder_regression(groups=args.groups, name='resnet18', norm=args.norm)
    # load pretrained
    #if args.best:
    #    model.load_state_dict(torch.load('./checkpoint/groups_20_lr_0.001_epoch_40_soft_label.pth'))  
    ckpt = torch.load('last.pth')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.encoder.load_state_dict(new_state_dict)
    
    # freeze the pretrained part
    #for (name, param) in model.encoder.named_parameters():
    #    param.requires_grad = False
    # 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer




def concate_feature_representation(current_z, current_y, previous_z, previous_y):
    z = torch.cat((current_z, previous_z), 0)
    y = torch.cat((current_y, previous_y), 0)
    return z, y


def test(model, test_loader):
    model.eval()
    #
    with torch.no_grad():
        for idx, (x, y, g) in enumerate(test_loader):
            bsz = x.shape[0]
            feature, label = [], []
            x, y = x.to(device), y.to(device)
            #
            _, z = model(x)
            #
            feature.extend(z.data.cpu().numpy())
            label.extend(y.data.cpu().numpy())
            #
    features = np.hstack(feature).reshape(-1,1)
    labels = np.hstack(label).reshape(-1,1)
    #         
    z_tensor, y_tensor = torch.Tensor(features), torch.Tensor(labels)
    #print(f' z_tesnot shape is {z_tensor.shape} y shape is {y_tensor.shape}')
    return z_tensor, y_tensor




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
    #
    acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, shot_pred, shot_pred_gt, gmean_gt, gmean_pred = test(
        model, test_loader, train_labels, args)
    results = [acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, gmean_gt, gmean_pred]



    
    
    
