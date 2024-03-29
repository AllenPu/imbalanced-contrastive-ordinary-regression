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
#from train import get_dataset


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--sigma', default=1.0, type=float)
parser.add_argument('--data_dir', type=str,
                    default='/home/rpu2/scratch/data/imbalanced-regression/imdb-wiki-dir/data', help='data directory')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--dataset', type=str, default='imdb_wiki',
                    choices=['imdb_wiki'], help='dataset name')
parser.add_argument('--group_mode', default='i_g', type=str,
                    help=' b_g is balanced group mode while i_g is imbalanced group mode')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--reweight', type=str, default=None,
                    help='weight : inv or sqrt_inv')
parser.add_argument('--names', type=str, required=True)

def draw_tsne(tsne_z_pred, tsne_g_pred, tsne_g_gt, args):
    # tsne_z_pred : the embedding 
    # tsne_g_pred : the predicted group
    # tsne_g_gt : the ground truth group
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_pred = tsne.fit_transform(tsne_z_pred)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne_pred[:, 0], X_tsne_pred[:, 1],
                    c=tsne_g_gt, label="t-SNE true label")
    plt.legend()
    plt.savefig(f'./images/tsne_x_gt_group_{args.groups}_true_label.png', dpi=120)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne_pred[:, 0], X_tsne_pred[:, 1],
                    c=tsne_g_pred, label="t-SNE pred label")
    plt.legend()
    plt.savefig(f'./images/tsne_x_pred_group_{args.groups}_pred_lael.png', dpi=120)
    

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
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #                            momentum=args.momentum, weight_decay=args.weight_decay)
    return model





if __name__ == '__main__':
    exec('from train import get_dataset')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_data_loader(args)
    #
    model = get_model(args)
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
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_hat = y_chunk[0], y_chunk[1]
            # draw tsne
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            tsne_z_pred = torch.cat((tsne_z_pred, z.data.cpu()), dim=0)
            #tsne_x_gt = torch.cat((tsne_x_gt, inputs.data.cpu()), dim=0)
            tsne_g_pred = torch.cat((tsne_g_pred, g_index.data.cpu()), dim=0)
            tsne_g_gt = torch.cat((tsne_g_gt, group.data.cpu()), dim=0)
    # draw tsne
    draw_tsne(tsne_z_pred, tsne_g_pred, tsne_g_gt, args)
