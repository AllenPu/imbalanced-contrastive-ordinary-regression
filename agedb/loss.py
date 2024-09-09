from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss

class LAloss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LAloss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        iota_list = tau * np.log(cls_probs)

        self.iota_list = torch.cuda.FloatTensor(iota_list)

    def forward(self, x, target):
        #print(" x shape is {} taegt shape is {} iota is {}" .format(x.shape, target.shape, self.iota_list))
        output = x + self.iota_list

        return F.cross_entropy(output, target, reduction='sum')
    


# own implementation over the contrastive
# a line by line implementaton (potential be useful for selective contrastive)
def Ranked_Contrastive_Loss(z, g, temp):
    bsz = z.shape[0]
    #
    z = F.normalize(z, dim=1)
    #
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    #
    device = "cuda:{}".format(sim_matrix.get_device())
    #
    #eye_mask = ~torch.eye(bsz, bsz, dtype = bool)
    #triu_mask = torch.triu(eye_mask, diagonal=1)
    #zeros = torch.zeros(bsz, bsz)
    #
    l1_matrix = torch.zeros(bsz, bsz)
    for i in range(bsz):
        for j in range(bsz):
            l1_matrix[i][j] = torch.abs(g[i] - g[j])
    #                        
    #slice = len(uni) - 1
    #
    loss = 0
    #
    for i in range(bsz):
        _, cnt = torch.unique(l1_matrix[i], return_counts=True, sorted=True)
        #
        srt = torch.argsort(l1_matrix[i], 0, descending=False)
        #index = cnt[0]
        if len(cnt) == 1:
            continue  # return self
        head = cnt[0].item()
        #
        for index in range(head, len(srt)):
            j = srt[index].item()
            nominator = torch.exp(sim_matrix[i][j]/temp)
            # find which cnt section is in
            for s in range(1, len(cnt)+1):
                if torch.sum(cnt[:s]).item() > index:
                    slice = s
                    break
            # how much shift in denominator
            deno_head = torch.sum(cnt[:slice-1])
            # the index of the denomnator in matrix
            deno_index = srt[deno_head:]
            #
            deno_index = deno_index.to(device)
            #
            denominator_matirx = torch.gather(
                sim_matrix[i], index=deno_index, dim=0)
            #
            denominator = torch.sum(torch.exp(denominator_matirx/temp), dim=0)
            #
            loss_partial = -torch.log(nominator/denominator)
            loss += loss_partial
        #
    return loss



def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    loss = torch.sum(loss)
    return loss



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
        #
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
