import torchvision.transforms as transforms
from datasets.IMDBWIKI import IMDBWIKI
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dirs = '/home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data'
datasets = 'imdb_wiki'
df = pd.read_csv(os.path.join(data_dirs, f"{datasets}.csv"))
df_train, df_val, df_test = df[df['split'] ==
                               'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
train_dataset = IMDBWIKI(data_dir=data_dirs, df=df_train,
                         img_size=224, split='train', group_num=10)
val_dataset = IMDBWIKI(data_dir=data_dirs, df=df_val,
                       img_size=224, split='val', group_num=10)
test_dataset = IMDBWIKI(data_dir=data_dirs, df=df_test,
                        img_size=224, split='test', group_num=10)
print(f'done dataset')
train_loader = DataLoader(train_dataset,    batch_size=256,
                          shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                         num_workers=8, pin_memory=True, drop_last=False)
print(f'done dataloader')

for idx, (inputs, targets, group) in enumerate(val_loader):
    inputs, targets, group = inputs.to(
        device), targets.to(device), group.to(device)
    if idx % 100 == 0:
        print(f' {idx} done in val')


for idx, (x, y, g, w) in enumerate(train_loader):
    x, y, g = x.to(device), y.to(device), g.to(device)
    if idx % 100 == 0:
        print(f' {idx} done in train')
