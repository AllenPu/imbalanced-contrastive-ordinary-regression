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
parser.add_argument('--norm', action='store_true')



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
    optimizer = torch.optim.SGD(model.regressor.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer



def train_epoch(model, train_loader, opt, args):
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()
    for e in tqdm(range(args.epoch)):
        for idx, (x, y, g) in enumerate(train_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            opt.zero_grad()
            y_output,  z = model(x)
            #
            y_ =  torch.chunk(y_output,2,dim=-1)
            g_hat, y_hat = y_[0], y_[1]
            y_pred = torch.gather(y_hat, dim=1, index=g.to(torch.int64)) 
            #
            if args.soft_label:
                g_soft_label = soft_labeling(g, args).to(device)
                loss_ce = SoftCrossEntropy(g_hat, g_soft_label)
                print(f' soft label loss is {loss_ce.item()}')
            elif args.ce:
                loss_ce = F.cross_entropy(g_hat, g.squeeze().long(), reduction='mean')
            elif args.la :
                loss_la = LAloss(group_list)
                loss_ce = loss_la(g_hat, g.squeeze().long())
            else:
                loss_ce = 0
                #print(f' ce loss is {loss_ce.item()}')
            #if torch.isnan(loss_ce):
            #    print(f' g_hat is {g_hat[:10]} g is {g[:10]} z is {z[:10]}')
            #    assert 1==0
            loss_mse = mse(y_pred, y)
            loss = loss_mse + loss_ce
            loss.backward()
            opt.step()
    return model



def test_group_acc(model, train_loader, prefix):
    model = Encoder_regression(groups=args.groups, name='resnet18')
    model = torch.load(f'./models/best_{prefix}.pth')
    model.eval()
    pred, labels = [], []
    for idx, (x, y, g) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        with torch.no_grad():
            y_output,  z = model(x)
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_pred = y_chunk[0], y_chunk[1]
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            pred.extend(g_index.data.cpu().numpy())
            labels.extend(g.data.cpu().numpy())
    pred = np.array(pred)
    labels = np.array(labels)
    np.save(f'./acc/pred{prefix}.npy', pred)
    np.save(f'./acc/labels{prefix}.npy', labels)






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
    model = train_epoch(model, train_loader, optimizer, args)
    #torch.save(model, f'./models/best_{prefix}.pth')
    acc_g_avg, acc_mae_gt_avg, acc_mae_pred_avg, shot_pred, shot_pred_gt, gmean_gt, gmean_pred = test(
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
    #
    #torch.save(model, f'./checkpoint/{store_name}.pth')


    
    
    
