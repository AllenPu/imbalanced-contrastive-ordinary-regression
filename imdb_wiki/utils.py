import torch
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import os
import random
import torch.nn as nn
softmax = nn.Softmax(dim=-1)
import torch.nn.functional as F
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def shot_metric(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    print(f' total length is {len(labels)}')
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(labels[labels == l]))
        l1_per_class.append(
            np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(
            np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    ma, md, fw = [], [] ,[]

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
            ma.append(i)
        elif train_class_count[i] < low_shot_thr:
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            md.append(i)
            #print(train_class_count[i])
            #print(l1_per_class[i])
            #print(l1_all_per_class[i])
        else:
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])
            fw.append(i)

    print(f'many_shot_l1 count  {np.sum(many_shot_cnt)}')
    print(f' many index {ma}')
    print(f'median_shot_l1 count  {np.sum(median_shot_cnt)}')
    print(f' md index {md}')
    print(f'low_shot_l1 count {np.sum(low_shot_cnt)}')
    print(f' few index {fw}')
    #
    shot_dict = defaultdict(dict)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


def shot_metric_cls(g_pred, g, train_labels, test_labels, many_shot_thr=100, low_shot_thr=20):
    #
    g_pred = np.hstack(g_pred)
    g = np.hstack(g)
    test_labels = np.hstack(test_labels)
    #
    train_class_count, test_class_count, test_acc_sum = [], [], []
    #
    for l in np.unique(train_labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(test_labels[test_labels == l]))
        #
        index = np.where(test_labels == l)[0]
        #
        acc_sum = 0
        #
        if len(index) == 0:
            test_acc_sum.append(0)
            #print(" index 0")
        else:
            for i in index:
                acc_sum += g_pred[i] == g[i]
            test_acc_sum.append(acc_sum)
        #print(l)
        #
    #print(" test acc sum is ", test_acc_sum)
    many_shot_cls, median_shot_cls, low_shot_cls = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []
    #
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_cls.append(test_acc_sum[i])
            many_shot_cnt.append(test_class_count[i])
        if train_class_count[i] < low_shot_thr:
            low_shot_cls.append(test_acc_sum[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_cls.append(test_acc_sum[i])
            median_shot_cnt.append(test_class_count[i])
    #
    shot_dict = defaultdict(dict)
    #
    shot_dict['many']['cls'] = 100 * \
        np.sum(many_shot_cls)/np.sum(many_shot_cnt)
    shot_dict['median']['cls'] = 100 * \
        np.sum(median_shot_cls)/np.sum(median_shot_cnt)
    shot_dict['low']['cls'] = 100 * np.sum(low_shot_cls)/np.sum(low_shot_cnt)
    #print(" many {} median {} low {} ".format(many_shot_cnt, median_shot_cnt, low_shot_cnt))

    return shot_dict


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabale = False


def balanced_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    mse_per_class, l1_per_class = [], []
    for l in np.unique(labels):
        mse_per_class.append(
            np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(
            np.mean(np.abs(preds[labels == l] - labels[labels == l])))

    mean_mse = sum(mse_per_class) / len(mse_per_class)
    mean_l1 = sum(l1_per_class) / len(l1_per_class)
    return mean_mse, mean_l1


def shot_metric_balanced(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(labels[labels == l]))
        l1_per_class.append(
            np.mean(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(
            np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            #print(train_class_count[i])
            #print(l1_per_class[i])
            #print(l1_all_per_class[i])
        else:
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / len(many_shot_l1)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / len(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / len(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


def soft_labeling(g, args, step = 1):
    groups = args.groups
    soft_group = []
    for i in g:
        label = int(i.item())
        soft_label = [0 for i in range(groups)]
        soft_label[int(label)] = groups-1
        for j in range(0, label):
            soft_label[j] = groups - step - (label-j)
        for j in range(1, groups-label):
            soft_label[j+label] = groups - step - j
        soft_group.append(soft_label)
    soft_groups = torch.Tensor(soft_group)
    soft_group = torch.clamp(soft_groups, 0, args.groups-1)
    soft_groups = softmax(soft_groups)
    #print(f' shape of soft groups {soft_groups.shape}')
    return soft_groups


def SoftCrossEntropy(inputs, target, reduction='sum'):
    #print(f' input shape is {inputs.shape}')
    log_likelihood = -F.log_softmax(inputs, dim=1)
    #print(f' log_likelihood is {log_likelihood.shape}')
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss



def diversity_loss_regressor(y_pred, g, args, total_dataset_length=100):
    #
    ranges = total_dataset_length / args.groups
    #
    centroid = (g*ranges + (g+1)*10)/2
    #
    mse = nn.MSELoss()(y_pred, centroid)
    #
    return -mse

def feature_diversity(z, g, args, total_dataset_length=100):
    #
    num_sample, feature_dim = z.shape[0], z.shape[1]
    #
    ranges = total_dataset_length / args.groups
    #
    u_value, u_index, u_counts = torch.unique(g, return_inverse=True, return_counts=True)
    center_f = torch.zeros([len(u_value), feature_dim]).cuda()
    u_index = u_index.squeeze()
    center_f.index_add_(0, u_index, z)
    u_counts = u_counts.unsqueeze(1)
    center_f = center_f / u_counts
    p = F.normalize(center_f, dim=1)
    _features_center = p[u_index, :]
    _features = F.normalize(z, dim=1)
    diverse_loss = torch.sum((_features - _features_center).pow(2),1)
    diverse_loss = torch.mean(diverse_loss)
    #
    return -diverse_loss


def topk_uncertain(y_out, g,  top_k = 3):
    y_out, g = y_out.cpu(), g.cpu()
    y_chunk = torch.chunk(y_out, 2, dim=1)
    #
    g_pred, y_pred = y_chunk[0], y_chunk[1]
    #
    _, k_g = g_pred.topk(top_k, dim=1, largest=True, sorted=True)
    #
    y_topk = torch.gather(y_pred, dim=1, index=k_g)
    #
    #y_top_k = torch.sum(y_topk, dim=-1)/top_k
    #
    g_hat = torch.argmax(g_pred, dim=1).unsqueeze(-1)
    y_hat = torch.gather(y_pred, dim=1, index=g_hat)
    #
    y_gt = torch.gather(y_pred, dim=1, index=g.to(torch.int64))
    #
    y_all = torch.cat((y_topk, k_g, y_hat, g_hat, y_gt, g), 1)
    #
    #y_all = torch.cat((y_topk, y_2), 1)
    #
    if os.path.exists('./y.gt'):
        y = torch.load('y.gt')
        y = torch.cat((y, y_all), 0)
        torch.save(y, 'y.gt')
    else:
        torch.save(y_all, 'y.gt')
    #
    if os.path.exists('./y_pred.gt'):
        y_ = torch.load('y_pred.gt')
        y_ = torch.cat((y_, y_pred), 0)
        torch.save(y_, 'y_pred.gt')
    else:
        torch.save(y_pred, 'y_pred.gt')





class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    


def validate(val_loader, model, regressor, train_labels=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            z = model(inputs)
            outputs = regressor(z)

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()


        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
        

def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    print(f' preds {preds.shape}, labels {labels.shape}')
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])
    


    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)
    
    return shot_dict
