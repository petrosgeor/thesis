import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = 'cuda'
EPS=1e-8

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
                loss_n = weights[n] * inner_n_log
                loss_p = torch.cat([loss_p, loss_n], dim=0)
            return -torch.mean(loss_p)




class KLClusterDivergance(nn.Module):
    def __init__(self):
        super(KLClusterDivergance, self).__init__()
        self.target_dist = torch.full((10, ), fill_value=1/10).to(device)
    
    def forward(self, probs: torch.Tensor):
        p = torch.mean(probs, dim=0)
        return torch.sum(p * (p/self.target_dist).log())




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


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight
        self.entropy = Entropy(small_number=EPS)

    def forward(self, anchors, neighbors):
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        entropy_loss = self.entropy(torch.mean(anchors_prob, dim=0), input_as_probabilities=True)

        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss
    

# z1 = torch.randn(1000, 10)
# z2 = torch.randn(1000, 10)
# loss1 = SCANLoss()
# loss2 = ClusterConsistencyLoss()

# _, t1, _ = loss1(z1, z2)

# x1 = F.softmax(z1, dim=1)
# x2 = F.softmax(z2, dim=1)

# t2 = loss2(x1, x2)


