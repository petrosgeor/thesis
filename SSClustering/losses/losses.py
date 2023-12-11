import torch
import torch.nn as nn
import numpy as np

device = 'cuda'


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

        mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        mask[n1, n2] = True
        mask[n2, n1] = True


        nominator_ij = Sim_matrix[n1, n2]
        nominator_ji = Sim_matrix[n2, n1]
        nominator = torch.cat((nominator_ij, nominator_ji), dim=0)
        denominator = (~mask * Sim_matrix).sum(dim=1)
        return -torch.mean((nominator/denominator).log())

class InfoNCELossEuclidean(nn.Module):
    def __init__(self, temperature):
        super(InfoNCELossEuclidean, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        representations = torch.vstack((z_i, z_j))
        Sim_matrix = -torch.cdist(representations, representations, p=2)/self.temperature
        Sim_matrix = torch.exp(Sim_matrix)

        n1 = torch.arange(0, batch_size)
        n2 = torch.arange(batch_size, 2*batch_size)

        mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=str(z_i.device))
        mask[n1, n2] = True
        mask[n2, n1] = True

        nominator_ij = Sim_matrix[n1, n2]
        nominator_ji = Sim_matrix[n2, n1]
        nominator = torch.cat((nominator_ij, nominator_ji), dim=0)
        denominator = (~mask * Sim_matrix).sum(dim=1)
        return -torch.mean((nominator/denominator).log())



class ClusterConsistencyLoss(nn.Module):
    def __init__(self, temperature = 0.2):
        super(ClusterConsistencyLoss, self).__init__()
        self.small_number = 1e-6
        self.temperature = temperature

    def forward(self, probs1: torch.Tensor, probs2: torch.Tensor, relations: torch.Tensor = None) -> torch.Tensor:
        if relations == None:
            inner_products = (probs1 * probs2).sum(dim=1)/self.temperature + self.small_number
            return -torch.mean(inner_products.log())
        elif relations is not None:
            inner_products = (probs1 * probs2).sum(dim=1)/self.temperature + self.small_number
            inner_products_log = inner_products.log()
            return -torch.mean(relations * inner_products_log)




class ClusterEntropyLoss(nn.Module):
    def __init__(self):
        super(ClusterEntropyLoss, self).__init__()
    def forward(self, probs: torch.Tensor):
        priors = torch.mean(probs, dim=0)
        return torch.sum(priors * priors.log())


class KLClusterDivergance(nn.Module):
    def __init__(self):
        super(KLClusterDivergance, self).__init__()
        self.target_dist = torch.full((10, ), fill_value=1/10).to(device)
    
    def forward(self, probs: torch.Tensor):
        p = torch.mean(probs, dim=0)
        return torch.sum(p * (p/self.target_dist).log())




