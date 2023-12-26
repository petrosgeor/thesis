import torch
import torch.nn as nn
from torch.nn import functional as F



class ClusterConsistencyLoss(nn.Module):
    def __init__(self, temperature = 0.2, threshold = -5):
        super(ClusterConsistencyLoss, self).__init__()
        self.small_number = 1e-6
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, probs1: torch.Tensor, probs2: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        if weights == None:
            #inner_products = (probs1 * probs2).sum(dim=1)/self.temperature + self.small_number
            inner_products = (probs1 * probs2).sum(dim=1)
            return -torch.mean(inner_products.log())
        elif weights is not None:
            p = torch.where(weights > 0)[0]
            n = torch.where(weights < 0)[0]
            # inner_products = (probs1 * probs2).sum(dim=1)
            # inner_products_log = inner_products.log()
            # return -torch.mean(weights * inner_products_log)
            inner_p = (probs1[p,:] * probs2[p,:]).sum(dim=1)
            inner_p_log = inner_p.log()
            loss_p = weights[p] * inner_p_log

            if n.numel() != 0:
                inner_n = (probs1[n,:] * probs2[n,:]).sum(dim=1)
                inner_n_log = (inner_n).log()
                inner_n_log[inner_n_log < self.threshold] = 0
                loss_n = 0.2*weights[n] * inner_n_log
                loss_p = torch.cat([loss_p, loss_n], dim=0)
            return -torch.mean(loss_p)

class Entropy(nn.Module):
    def __init__(self, small_number: float):
        super(Entropy, self).__init__()
        self.small_number = small_number
    
    def forward(self, x, input_as_probabilities):
        if input_as_probabilities:
            x_ = torch.clamp(x, min=self.small_number)
            b = x_ * torch.log(x_)
        
        else:
            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

        if len(b.size()) == 2:
            return -b.sum(dim=1).mean()
        elif len(b.size()) == 1:
            return -b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

