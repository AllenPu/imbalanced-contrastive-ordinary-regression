import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from collections import defaultdict
from scipy.stats import gmean
import random
from torch.distributions import Categorical, kl
import torch.nn as nn
softmax = nn.Softmax(dim=-1)
import torch.nn.functional as F
from utils import *


class AverageMeter(object):
    def __init__(self,  name = '', fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def shot_metric(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
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
            np.sum(np.abs(preds[labels == l] - labels[labels == l])))
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
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))

    mean_mse = sum(mse_per_class) / len(mse_per_class)
    mean_l1 = sum(l1_per_class) / len(l1_per_class)
    return mean_mse, mean_l1


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
    #shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / len(median_shot_cnt)
    #shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / len(low_shot_cnt)
    #shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


# calculate the entropy
def cal_entropy(output, g, topk=3, mode = 'train'):
    # output is the output prob
    # g is the ground truth label
    if mode == 'train':
        sort = torch.sort(g, dim=-1)
    else:
        sort = torch.sort(output, dim=-1)
    p_sort = sort.indices[:, : topk]
    p = torch.gather(output, dim=-1, index=p_sort)
    p_ent = Categorical(p).entropy()
    p_entropy = torch.sum(p_ent)
    #
    return p_entropy


# for infer in test only
def cal_ensemble_reg(output_cls, output_reg, args, topk=3, mode = 'train'):
    #
    row, col, groups = output_cls.shape[0], output_cls.shape[1], args.groups-1
    #
    largest_prob = torch.topk(output_cls, 1, largest=True)
    #
    topk_index = largest_prob.repeat(1,topk)
    #
    shift = torch.Tensor([-1,0,1]).repeat(row, 1).to(torch.int64)
    #
    ens_index = topk_index + shift
    ens_index = torch.clamp(ens_index,0, groups)
    #
    ens_reg = torch.gather(output_reg, dim=1, index=ens_index)
    #
    ens_cls = torch.gather(output_cls, dim=1, index=ens_index)
    #
    #softmax = nn.Softmax(dim=-1)
    #
    ens_cls_prob = softmax(ens_cls)
    #
    reg = torch.sum(torch.matmul(ens_reg, ens_cls_prob))
    #
    return reg


#
def soft_labeling(g, args):
    step = args.step
    groups = args.groups
    soft_group = []
    for i in g:
        label = int(i.item())
        soft_label = [0 for i in range(groups)]
        soft_label[int(label)] = args.scale*(groups-1)
        for j in range(0, label):
            soft_label[j] = (1/args.scale)*(groups - step -  (label-j))
        for j in range(1, groups-label):
            soft_label[j+label] = (1/args.scale)*(groups - step - j)
        soft_group.append(soft_label)
    soft_groups = torch.Tensor(soft_group)
    soft_groups =  torch.clamp(soft_groups, 0, groups-1)
    soft_groups = softmax(soft_groups)
    return soft_groups


def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


#
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

    # return torch is, [biggest group prediction, second group prediction, third group prediction, index1, index2, index3,  y_pred, g_pred, y_gt, g_gt]


# this is implemented on the y output
# should be on the side of representation
def diversity_loss(y_pred, g, args, total_dataset_length=100):
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




    
def shot_count(train_labels, many_shot_thr=100, low_shot_thr=20):
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count = []
    #
    maj_class, med_class, min_class = [], [], []
    #
    for l in np.unique(train_labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
    #
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            maj_class.append(np.unique(train_labels)[i])
        elif train_class_count[i] < low_shot_thr:
            min_class.append(np.unique(train_labels)[i])
        else:
            med_class.append(np.unique(train_labels)[i]) 
    #
    return maj_class, med_class, min_class


def shot_reg(label, pred, maj, med, min):
    # how many preditions in this shots
    pred_dict = {'maj':0, 'med':0, 'min':0}
    # how many preditions from min to med, min to maj, med to maj, min to med
    pred_label_dict = {'min to med':0, 'min to maj':0, 'med to maj':0, 'med to min':0, 'maj to min':0, 'maj to med':0}
    #
    pred = int_tensors(pred)
    #
    #print(maj)
    #print(med)
    #print(min)
    #print(f' size of label {type(label)} size of pred {type(pred)}')
    #
    labels, preds = np.stack(label), np.stack(pred)
    #
    #print(f' labels {labels[:100]} preds {preds[:100]}')
    #dis = np.floor(np.abs(labels - preds)).tolist()
    bsz = labels.shape[0]
    for i in range(bsz):
        k_pred = check_shot(preds[i],maj, med, min)
        k_label = check_shot(labels[i],maj, med, min)
        if k_pred in pred_dict.keys():
            pred_dict[k_pred] = pred_dict[k_pred] + 1
        pred_shift = check_pred_shift(k_pred, k_label)
        if pred_shift in pred_label_dict.keys():
            pred_label_dict[pred_shift] = pred_label_dict[pred_shift] + 1
    return pred_dict['maj'], pred_dict['med'], pred_dict['min'], \
        pred_label_dict['min to med'], pred_label_dict['min to maj'], pred_label_dict['med to maj'],pred_label_dict['med to min'],pred_label_dict['maj to min'],pred_label_dict['maj to med']


def check_shot(e, maj, med, min):
    if e in maj:
        return 'maj'
    elif e in med:
        return 'med'
    else:
        return 'min'
    
# check reditions from min to med, min to maj, med to maj
def check_pred_shift(k_pred, k_label):
    if k_pred is 'med' and k_label is 'min':
        return 'min to med'
    elif k_pred is 'maj' and k_label is 'min':
        return 'min to maj'
    elif k_pred is 'maj' and k_label is 'med':
        return 'med to maj'
    elif k_pred is 'min' and k_label is 'med':
        return 'med to min'
    elif k_pred is 'min' and k_label is 'maj':
        return 'maj to min'
    elif k_pred is 'med' and k_label is 'maj':
        return 'maj to med'
    else:
        return 'others'
      

def int_tensors(pred):
    pred = torch.Tensor(pred)
    #pred = pred - torch.floor(pred)
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    diff = pred - torch.floor(pred)
    diff = torch.where(diff > 0.5, one, diff)
    diff = torch.where(diff < 0.5, zero, diff)
    pred = torch.floor(pred) + diff
    pred = torch.clamp(pred, 0, 100)
    pred = pred.tolist()
    return pred


def cal_frob_norm(y, feat, majs, meds, mino, maj_shot, med_shot, min_shot, maj_shot_nuc, med_shot_nuc, min_shot_nuc, device):
    bsz = y.shape[0]
    # calculate the frob norm of test on different shots
    maj_index, med_index, min_index = [], [], []
    for i in range(bsz):
        if y[i] in majs:
            maj_index.append(i)
        elif y[i] in meds:
            med_index.append(i)
        else:
            min_index.append(i)
    #
    if len(maj_index) != 0:
        majority = torch.index_select(feat, dim=0, index=torch.LongTensor(maj_index).to(device))
        ma = torch.mean(torch.norm(majority, p='fro', dim=-1))
        ma_nuc = torch.norm(majority, p='nuc')/majority.shape[0]
        #maj_shot = math.sqrt(maj_shot**2 + ma)
        maj_shot.update(ma.item(), majority.shape[0])
        maj_shot_nuc.update(ma_nuc.item(), majority.shape[0])
    if len(med_index) != 0:
        median = torch.index_select(feat, dim=0, index=torch.LongTensor(med_index).to(device))
        md = torch.mean(torch.norm(median, p='fro', dim=-1))
        md_nuc = torch.norm(median, p='nuc')/median.shape[0]
        #med_shot = math.sqrt(med_shot**2 + md)
        med_shot.update(md.item(), median.shape[0])
        med_shot_nuc.update(md_nuc.item(), median.shape[0])
    if len(min_index) != 0:
        minority = torch.index_select(feat, dim=0, index=torch.LongTensor(min_index).to(device))
        mi = torch.mean(torch.norm(minority, p='fro', dim=-1))
        mi_nuc = torch.norm(minority, p='nuc')/minority.shape[0]
        #min_shot = math.sqrt(mi**2 + mi)
        min_shot.update(mi.item(), minority.shape[0])
        min_shot_nuc.update(mi_nuc.item(), minority.shape[0])
    return maj_shot, med_shot, min_shot, maj_shot_nuc, med_shot_nuc, min_shot_nuc
   