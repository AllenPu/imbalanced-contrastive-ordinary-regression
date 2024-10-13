import pandas as pd
import os
import torch
import time
import argparse
from tqdm import tqdm
import pandas as pd
#from network import *
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
# margin of the groups , .e.g. for a 10 groups label, step = 2 [...,6,8,10,8,6,...]
parser.add_argument('--step', type=float, default=1)
parser.add_argument('--la', action='store_true')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--best', action='store_true')
#
parser.add_argument('--asymm', action='store_true', help='if use the asymmetric soft label')
parser.add_argument('--single', action='store_true', help='if single output')
parser.add_argument('--fine_tune', action='store_false')





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
    if args.single:
        model = Encoder_regression_single(name='resnet18')
    else:
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
    # SFT
    if args.fine_tune:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # Linear prob
    else:
        for (name, param) in model.encoder.named_parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(model.regressor.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer


# train the model with single output
def train_epoch_single(model, train_loader, opt, args):
    model = model.to(device)
    model.train()
    mse = nn.MSELoss()
    for e in tqdm(range(args.epoch)):
        for idx, (x, y, g) in enumerate(train_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            opt.zero_grad()
            y_pred, _ = model(x)
            #print(y_pred.shape)
            loss = mse(y_pred, y)
            loss.backward()
            opt.step()
    return model


# train the model with the multiple experts
def train_epoch(model, train_loader, opt, args):
    model = model.to(device)
    mse = nn.MSELoss()
    for e in tqdm(range(args.epoch)):
        model.train()
        for idx, (x, y, g) in enumerate(train_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            #
            bsz = x.shape[0]
            #
            opt.zero_grad()
            y_output,  z = model(x)
            #
            y_ =  torch.chunk(y_output,2,dim=-1)
            #print(f' y shape {y_output.shape}')
            g_hat, y_hat = y_[0], y_[1]
            y_pred = torch.gather(y_hat, dim=1, index=g.to(torch.int64)) 
            #
            if args.soft_label:
                g_soft_label = soft_labeling(g, args).to(device)
                # rescale the soft label
                if args.asymm:
                    g_soft_label = asymmetric_soft_labeling(group_list, g_soft_label)
                    #
                loss_ce = SoftCrossEntropy(g_hat, g_soft_label)
                #print(f' soft label loss is {loss_ce.item()}')
            elif args.ce:
                loss_ce = F.cross_entropy(g_hat, g.squeeze().long(), reduction='sum')
            elif args.la :
                loss_la = LAloss(group_list)
                loss_ce = loss_la(g_hat, g.squeeze().long())
            else:
                loss_ce = 0
                #print(f' ce loss is {loss_ce.item()}')
            #if torch.isnan(loss_ce):
            #    print(f' g_hat is {g_hat[:10]} g is {g[:10]} z is {z[:10]}')
            #    assert 1==0
            # print(f' loss ce is  {loss_ce.item()}')
            loss_mse = mse(y_pred, y)
            loss = loss_mse + loss_ce
            loss.backward()
            opt.step()
        #if e% 10 == 0:
        #    print(f' e peoch at {e}')
        #    test_multiple(model, test_loader, train_labels, args)
    return model


# no currently used
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




# test the single output model
def test_single(model, test_loader, train_labels, args):
    model.eval()
    test_mae_pred = AverageMeter()
    preds, label, gmeans = [], [], []
    criterion_gmean = nn.L1Loss(reduction='none')
    #
    for idx, (x,y,g) in enumerate(test_loader):
        with torch.no_grad():
            bsz = x.shape[0]
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred, _ = model(x)
            test_mae = F.l1_loss(pred, y)
            preds.extend(pred.cpu().numpy())
            label.extend(y.cpu().numpy())
            test_mae_pred.update(test_mae,bsz)
            #
            loss_gmean = criterion_gmean(pred, y)
            gmeans.extend(loss_gmean.cpu().numpy())
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



# test the multi output model
def test_multiple(model, test_loader, train_labels, args):
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


def asymmetric_soft_labeling(group_list, g_soft_label):
    bsz = g_soft_label.shape[0]
    total_num = sum(group_list)
    rescale_groups = [1-i/total_num for i in group_list]
    rescale_tensor = torch.Tensor(rescale_groups).repeat(bsz, 1).to(device)
    ##
    mask_1 = (g_soft_label == g_soft_label.max(dim=1, keepdim=True)[0])
    # remove all current group index by multiple 0 (leave the max soft label then process others)
    remove_1 = torch.where(mask_1, 1.0, 0.0)
    remove_group_soft = g_soft_label * remove_1 
    # remove all non current group index by multiple 0, reverse from above
    remove_2 = torch.where(mask_1, 0.0, 1.0)
    remove_non_group_soft = g_soft_label * remove_2 * rescale_tensor
    # final is the cumulative of both 
    g_soft_label = remove_non_group_soft + remove_group_soft
    return g_soft_label



if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader, group_list, train_labels = get_data_loader(args)
    model, optimizer = get_model(args)
    store_name = 'groups_' + str(args.groups) + '_lr_' + str(args.lr) + '_epoch_' + str(args.epoch)
    if args.soft_label:
        prefix = '_soft_label'
        if args.asymm:
            prefix = prefix + '_asymm'
        else:
            prefix = prefix + '_symm'
    elif args.la:
        prefix = '_la'
    elif args.ce:
        prefix = '_ce'
    else:
        print(f'no classification criterion specified !!!')
        prefix = '_no_cls'
    ################################
    if args.single:
        prefix = prefix + '_single'
    else:
        prefix = prefix + '_multi_expert'
    store_name = store_name + prefix
    #encoder, regressor = train_regressor(train_loader, model.encoder, model.regressor, optimizer, args)
    #validate(val_loader, encoder, regressor, train_labels=train_labels)
    print(f' Start to train !')
    #torch.save(model, f'./models/best_{prefix}.pth')
    
    if args.single:
        model = train_epoch_single(model, train_loader, optimizer, args)
        test_single(model, test_loader, train_labels, args)
    else:
        model = train_epoch(model, train_loader, optimizer, args)
        test_multiple(model, test_loader, train_labels, args)
    
    #
    #
    print(f' store name is {store_name}')
    #
    #torch.save(model, f'./checkpoint/{store_name}.pth')
    regressor_weight = model.regressor[0].weight.data
    name = ''
    if args.fine_tune:
        name = name + 'sft_'
    if args.soft_label and not args.asymm:
        name = name + 'symm_'
    if args.soft_label and args.asymm:
        name = name + 'asymm_'
    if not args.fine_tune:
        name = name + 'linear_prob_'
    #
    print(f'store name is {name}')
    #torch.save(regressor_weight, f'./{name}_weight.pt')

    
    
    
