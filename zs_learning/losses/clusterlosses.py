import torch
import torch.nn as nn
from torch.nn import functional as F


'https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/losses/losses.py'



device = 'cuda'
EPS=1e-8





class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss

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