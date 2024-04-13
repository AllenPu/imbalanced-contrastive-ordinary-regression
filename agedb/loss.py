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
