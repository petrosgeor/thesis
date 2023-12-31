import torch
import platform
from models import *
from lightly.models import ResNetGenerator
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment


def set_AwA2_dataset_path():
    system = platform.system()
    assert (system == 'Windows') | (system == 'Linux')
    if system == 'Windows':
        path = 'C:\\Users\Peter\PycharmProjects\Thesis\zs_learning\Animals_with_Attributes2'
    elif system == 'Linux':
        path = '/gpu-data/pger/Animals_with_Attributes2'
    return path


def find_indices_of_closest_embeddings(embedings: torch.Tensor, n_neighbors: int = 20) -> torch.Tensor:
    D = torch.matmul(embedings, embedings.T)
    indices = torch.topk(D, k=n_neighbors, dim=1)[1]
    return indices

def load_pretrained_backbone() -> tuple:
    '''x = resnet18()
    backbone = x['backbone']
    dim = x['dim']
    backbone.load_state_dict(torch.load('NeuralNets/moco_pretrained_backbone.pth'))
    return backbone, dim'''
    resnet = ResNetGenerator('resnet-18', 1)
    backbone = nn.Sequential(
                            *list(resnet.children())[:-1],
                             nn.AdaptiveAvgPool2d(1),
                            )
    backbone.load_state_dict(torch.load('NeuralNets/moco_pretrained_backbone.pth'))
    return backbone, 512

def initialize_clustering_net(n_classes: int=50, nheads: int=1):        # returns an instance of the clusteringNet class
    backbone, dim = load_pretrained_backbone()
    clusternet = ClusteringModel(backbone= {'backbone': backbone, 'dim': dim}, nheads=nheads, nclusters=n_classes)
    return clusternet


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0


def find_permutation_matrix(cost_matrix: torch.Tensor):
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1.
    return torch.from_numpy(P)