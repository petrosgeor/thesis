import torch
import platform
from models import *
from lightly.models import ResNetGenerator
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
import socket
import matplotlib.pyplot as plt


def set_AwA2_dataset_path():
    system = platform.system()
    assert (system == 'Windows') | (system == 'Linux')
    if system == 'Windows':
        path = 'C:\\Users\Peter\PycharmProjects\Thesis\zs_learning\Animals_with_Attributes2'
    elif system == 'Linux':
        hostname = socket.gethostname()
        if hostname == 'halki':
            path = '/gpu-data/pger/Animals_with_Attributes2'
        elif hostname == 'kalymnos':
            path = '/gpu-data2/pger/Animals_with_Attributes2'
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

def load_scan_trained_model(n_classes: int=50): # loads the model trained using SCAN    
    x = resnet18()
    resnet = x['backbone']
    resnet.load_state_dict(torch.load('NeuralNets/backbone_AwA2.pth'))
    clusternet = ClusteringModel(backbone={'backbone':resnet, 'dim':512}, nclusters=n_classes)
    clusternet.cluster_head.load_state_dict(torch.load('NeuralNets/cluster_head_AwA2.pth'))
    return clusternet.to('cpu')

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


def FreezeResnet(model):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    return model



def visualize_embeddings(embeddings: np.ndarray, labels: np.ndarray, means: np.ndarray, Ids: np.ndarray, path2save: str):   # use this function to visualize 2d embeddings and the projected semantic vectors
    unique_labels = np.unique(labels)

    if len(unique_labels) == 40:
        colors = plt.cm.tab20.colors[:20] + plt.cm.tab20b.colors[:20]
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]
    elif len(unique_labels) == 50:
        colors = colors = (plt.cm.tab20.colors[:20] + plt.cm.tab20b.colors[:20] + plt.cm.tab20c.colors[:10])
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]
    elif len(unique_labels) == 10:
        colors = plt.cm.tab10.colors[:10]
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]

    plt.figure()
    for i, color in zip(unique_labels, hex_colors):
        plt.scatter(embeddings[labels == i, 0], embeddings[labels == i, 1], c=[color], label=f'Label {i}', s=10, alpha=0.5)

    if means.any() != None:
        for i, (x, y) in enumerate(means):
            plt.scatter(x, y, s=50, color='black')
            plt.text(x, y, str(Ids[i]), fontsize=9, ha='right')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Samples by Label')
    plt.savefig(path2save)