import os
import pandas as pd
from torch.utils.data import DataLoader
from datasets.agedb import *
import torch
from network import *
import argparse
from utils import topk_uncertain

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
parser.add_argument('--model_name', type=str, default='',
                    help='name of the model')
parser.add_argument('--model_depth', type=int, default=50,
                    help='depth of the model')
parser.add_argument('--sigma', type=int, default=1)



data_dir = '/home/ruizhipu/scratch/regression/imbalanced-regression/agedb-dir/data'
img_size = 224
reweight = None
groups = 10
batch_size = 128
workers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loader():
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] =='train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    #train_labels = df_train['age']
    #train_dataset = AgeDB(data_dir=data_dir, df=df_train, img_size=img_size,
    #                      split='train', reweight=reweight, group_num=groups)
    #val_dataset = AgeDB(data_dir=data_dir, df=df_val,
    #                    img_size=img_size, split='val', group_num=groups)
    test_dataset = AgeDB(data_dir=data_dir, df=df_test,
                         img_size=img_size, split='test', group_num=groups)
    #group_list = train_dataset.get_group_list()
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                          num_workers=workers, pin_memory=True, drop_last=False)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    #                        num_workers=workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True, drop_last=False)
    #print(f"Training data size: {len(train_dataset)}")
    #print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return test_loader



def load_model(args):
    model = ResNet_regression(args).to(device)
    model.load_state_dict(torch.load(args.model_name), False)
    print(model)
    return model


def forwards(model, loader):
    with torch.no_grad():
        for idx, (x, y, g) in enumerate(loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
            y_output, _ = model(x)
            topk_uncertain(y, g)





if __name__ == '__main__':
    args = parser.parse_args()
    model = load_model(args)
    test_loader = get_data_loader()
    forwards(model, test_loader)
