import torch
import platform
from models import *
from lightly.models import ResNetGenerator

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




# clusternet = initialize_clustering_net()
# x = torch.randn(10, 3, 64, 64)