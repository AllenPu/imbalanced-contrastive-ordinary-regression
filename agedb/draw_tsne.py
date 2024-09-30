from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import torch
from datasets.agedb import AgeDB
from model import *
import pandas as pd
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
import numpy as np
#from train import get_dataset


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--seed', default=3407)
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
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
parser.add_argument('--model_name', type=str, help='path to the pretrained model')
parser.add_argument('--store_name', type=str, help='path to the pretrained model', default='none')
parser.add_argument('--single', action='store_true', help='if single output')
#parser.add_argument('--names', type=str, required=True, help='names of the draw picture')

def draw_tsne(tsne_z_pred, tsne_g_pred, tsne_g_gt, args):
    # tsne_z_pred : the embedding 
    # tsne_g_pred : the predicted group
    # tsne_g_gt : the ground truth group
    if args.soft_label:
        prefix = 'soft_label'
        if args.asymm:
            prefix = prefix + '_asymm'
        else:
            prefix = prefix + '_symm'
    elif args.ce:
        prefix = 'ce'
    elif args.la :
        prefix = 'la'
    else:
        prefix = 'none'
        print(" no prefix")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_pred = tsne.fit_transform(tsne_z_pred)
    #
    model_name = args.store_name
    #
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne_pred[:, 0], X_tsne_pred[:, 1],
                    c=tsne_g_gt, label="t-SNE true label")
    plt.legend()
    #plt.savefig(f'./images/tsne_x_gt_group_{args.groups}_epoch_{args.epoch}_{prefix}__true_label.png', dpi=120)
    plt.savefig(f'./images/{model_name}_gt.png', dpi=300)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne_pred[:, 0], X_tsne_pred[:, 1],
                    c=tsne_g_pred, label="t-SNE pred label")
    plt.legend()
    #plt.savefig(f'./images/tsne_x_pred_group_{args.groups}_epoch_{args.epoch}_{prefix}_pred_label.png', dpi=120)
    plt.savefig(f'./images/{model_name}_pred.png', dpi=300)
    plt.close()



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
        model = Encoder_regression(groups=args.groups, name='resnet18')
    # load pretrained
    #if args.best:
    #    model.load_state_dict(torch.load('./checkpoint/groups_20_lr_0.001_epoch_40_soft_label.pth'))  
    ckpt = torch.load(args.model_name)
    new_state_dict = OrderedDict()
    try:
        for k,v in ckpt['model'].items():
            key = k.replace('module.','')
            keys = key.replace('encoder.','')
            new_state_dict[keys] =  v
        model.encoder.load_state_dict(new_state_dict)
    except:
        model = ckpt
        print('load ckpt as model')
    
    # freeze the pretrained part
    #for (name, param) in model.encoder.named_parameters():
    #    param.requires_grad = False
    # 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer


# draw the absolute difference between different classifciation criteria
def draw_aboslute_difference():
    labels_ce = np.load('./acc/labels_ce.npy')
    labels_la = np.load('./acc/labels_la.npy')
    labels_soft_label = np.load('./acc/labels_soft_label.npy')
    pred_ce = np.load('./acc/pred_ce.npy')
    pred_la = np.load('./acc/pred_la.npy')
    pred_soft_label = np.load('./acc/pred_soft_label.npy')


    length = labels_ce.shape[0]
    diff_ce = {}
    diff_la = {}
    diff_soft = {}
    for i in range(length):
        soft_ab = int(np.abs(pred_soft_label[i]-labels_soft_label[i]))
        diff_soft[soft_ab] = diff_soft.get(soft_ab, 0) + 1
        ce_ab = int(np.abs(pred_ce[i]-labels_ce[i]))
        diff_ce[ce_ab] = diff_ce.get(ce_ab, 0) + 1
        la_ab = int(np.abs(pred_la[i]-labels_la[i]))
        diff_la[la_ab] = diff_la.get(la_ab, 0) + 1
    print(f'soft ab {diff_soft} len : {len(diff_soft)}')
    print(f'ce ab {diff_ce} len : {len(diff_ce)}')
    print(f'la ab {diff_la} len : {len(diff_la)}')
    list_soft = [diff_soft[key] for key in sorted(diff_soft)]
    list_ce = [diff_ce[key] for key in sorted(diff_ce)]
    list_la = [diff_la[key] for key in sorted(diff_la)]



    #colors = ['r', 'g', 'b']
    #yticks = ['soft','ce', 'la']
    df = pd.DataFrame({'soft': np.array(list_soft), 'ce': np.array(list_ce), 'la':np.array(list_la)}, columns=['soft', 'ce','la'])
    df.plot.hist(alpha=0.8,bins=length)
    plt.savefig('diff.jpg')
    #plt.show()




if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_data_loader(args)
    #
    model, _ = get_model(args)
    model = model.to(device)
    tsne_z_pred = torch.Tensor(0)
    tsne_g_pred = torch.Tensor(0)
    tsne_g_gt = torch.Tensor(0)
    model.eval()
    for idx, (inputs, targets, group) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        group = group.to(device)
        #
        with torch.no_grad():
            y_output, z = model(inputs.to(torch.float32))
            if not args.single:
                y_chunk = torch.chunk(y_output, 2, dim=1)
                g_hat, y_hat = y_chunk[0], y_chunk[1]
                # draw tsne
                g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            else:
                y_pred = torch.floor(y_output)
                g_index = y_pred
            tsne_z_pred = torch.cat((tsne_z_pred, z.data.cpu()), dim=0)
            #tsne_x_gt = torch.cat((tsne_x_gt, inputs.data.cpu()), dim=0)
            tsne_g_pred = torch.cat((tsne_g_pred, g_index.data.cpu()), dim=0)
            #y_pred = torch.floor(y_output)
            tsne_g_gt = torch.cat((tsne_g_gt, targets.data.cpu()), dim=0)
    # draw tsne
    draw_tsne(tsne_z_pred, tsne_g_pred, tsne_g_gt, args)
