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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" training on ", device)
parser = argparse.ArgumentParser('argument for training')





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



def get_model(args, groups):
    model = Encoder_regression(groups=groups, name='resnet18')
    # load pretrained
    ckpt = torch.load('last.pth')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys]=v
    model.encoder.load_state_dict(new_state_dict)
    # freeze the pretrained part
    for (name, param) in model.encoder.named_parameters():
        param.requires_grad = False
    #
    optimizer = torch.optim.SGD(model.regressor.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer



def train(model, train_loader, opt, args):
    mse = nn.MSELoss()
    for e in tqdm(range(args.epoch)):
        for idx, (x, y, g) in enumerate(train_loader):
            x, y, g = x.to(device), y.to(device), g.to(device)
            opt.zero_grad()
            y_output, z = model(x)
            y =  torch.chunk(y_output,2,dim=-1)
            g_hat, y_hat = y[0], y[1]
            y_pred = torch.gather(y_hat, dim=1, index=g.to(torch.int64))
            g_soft_label = soft_labeling(g, args).to(device)
            loss_ce_soft = SoftCrossEntropy(g_hat, g_soft_label)
            loss_mse = mse(y_pred, y)
            loss = loss_mse + loss_ce_soft
            loss.backward()
            opt.step()
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    