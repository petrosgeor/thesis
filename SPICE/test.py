
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import * 


n_clusters = 20
backbone = resnet18()
contrastive_model = ContrastiveModel(backbone=backbone)

cluster_model = ClusteringModel(backbone={'backbone': contrastive_model.backbone, 'dim': contrastive_model.backbone_dim}, nclusters=20)

e = torch.load('NeuralNets/scan_cifar20.pth')
cluster_model.load_state_dict(e)






















