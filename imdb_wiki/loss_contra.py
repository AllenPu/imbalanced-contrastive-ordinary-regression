import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cos_sim':
            return - F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        #features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        features = F.normalize(features, dim = -1)
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
    

class Softlabel_Loss(nn.Module):
    def __init__(self, label_smooth=None, class_num=10):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        # cross entropy loss with label smoothing
        logprobs = F.log_softmax(pred, dim=1)	# softmax + log
        target = F.one_hot(target, self.class_num)	# 转换成one-hot
            
        # label smoothing
        # 实现 1
        # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num 	
        # 实现 2
        # implement 2
        target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
        loss = -1*torch.sum(target*logprobs, 1)
        return loss.mean()
    

class RnCLoss_pairwise(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_pairwise, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()
            if math.isnan(loss.item()):
                print(f'pos_log_probs is {pos_log_probs} k is {k}')
                assert 1 == 2
        return loss