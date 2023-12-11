import torch
import torch.nn as nn
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor)-> torch.Tensor:
        '''
        :param z_i: The first view of a batch
        :param z_j: The second view of a batch
        :return: InfoNCE Loss
        '''
        batch_size = z_i.shape[0]
        representations = torch.vstack((z_i, z_j))
        Sim_matrix = torch.matmul(representations, representations.T)/self.temperature
        Sim_matrix = torch.exp(Sim_matrix)

        n1 = torch.arange(0, batch_size)
        n2 = torch.arange(batch_size, 2*batch_size)

        mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool)
        mask[n1, n2] = True
        mask[n2, n1] = True


        nominator_ij = Sim_matrix[n1, n2]
        nominator_ji = Sim_matrix[n2, n1]
        nominator = torch.cat((nominator_ij, nominator_ji), dim=0)
        denominator = (~mask * Sim_matrix).sum(dim=1)
        return -torch.mean((nominator/denominator).log())




class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1), 0)
        mask = mask * logits_mask
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

lossss = SimCLRLoss(temperature=0.5)
x = torch.randn(100, 2, 128)
x = x + 5
x = F.normalize(x, dim=2)
print(lossss(x))

Infonce = InfoNCELoss(temperature=0.5)

z1 = x[:, 0]
z2 = x[:, 1]
print(Infonce(z1, z2))
