import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda'
EPS=1e-8



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