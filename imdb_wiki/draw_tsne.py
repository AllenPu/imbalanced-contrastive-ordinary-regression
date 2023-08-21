from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import torch
from network import ResNet_regression
from datasets.IMDBWIKI import IMDBWIKI
from train import get_dataset


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
    plt.savefig('./images/tsne_x_pred_{}_sigma_{}_group_{}_model_{}_true_label.png'.format(
        args.lr, args.sigma, args.groups, args.model_depth), dpi=120)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne_pred[:, 0], X_tsne_pred[:, 1],
                    c=tsne_g_pred, label="t-SNE pred label")
    plt.legend()
    plt.savefig('./images/tsne_x_pred_{}_sigma_{}_group_{}_model_{}_pred_lael.png'.format(
        args.lr, args.sigma, args.groups, args.model_depth), dpi=120)
    


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader,  cls_num_list, train_labels = get_dataset(
        args)
    model = ResNet_regression(args).to(device)
    model.load_state_dict(torch.load(
        './models/model_{}.pth'.format(args.names)))
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
