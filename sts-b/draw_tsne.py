from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
    