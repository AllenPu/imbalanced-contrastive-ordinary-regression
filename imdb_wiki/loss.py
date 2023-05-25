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

        return F.cross_entropy(output, target)




class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(
            torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss


def Ranked_Contrastive_Loss(z, g, temp):
    bsz = z.shape[0]
    #
    z = F.normalize(z, dim = 1)
    #
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim = -1)
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
    for i in range(bsz):
        _, cnt = torch.unique(l1_matrix[i], return_counts = True, sorted = True)
        #
        srt = torch.argsort(l1_matrix[i], 0, descending=False)
        #index = cnt[0]
        if len(cnt) == 1:
            continue # return self
        head = cnt[0].item()
        #
        loss = 0
        #
        for index in range(head, len(srt)):
            j = srt[index].item()
            nominator = torch.exp(sim_matrix[i][j]/temp)
            # find which cnt section is in
            for s in range(1,len(cnt)+1):
                if torch.sum(cnt[:s]).item() > index:
                    slice = s
                    break
            # how much shift in denominator 
            deno_head = torch.sum(cnt[:slice-1])
            # the index of the denomnator in matrix
            deno_index = srt[deno_head : ]
            # 
            denominator_matirx = torch.gather(sim_matrix[i], index=deno_index, dim=0)
            #
            denominator = torch.sum(torch.exp(denominator_matirx/temp), dim=0)
            #
            loss_partial = -torch.log(nominator/denominator)
            loss += loss_partial
        #
        return loss
                
                
            

