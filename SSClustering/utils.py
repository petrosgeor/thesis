import os
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.manifold import TSNE, MDS
from models import *

device = 'cuda'

def running_on_colab():
    import os
    return 'COLAB_GPU' in os.environ


def find_environment() -> str:

    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/CIFAR10/'
    else:
        path = 'CIFAR10/'
    return path









class SoftNNLossNeg(nn.Module):     # Use if an image has only negatives in the A_matrix
    def __init__(self, temperature: float, w: float = 2) -> None:
        super(SoftNNLossNeg, self).__init__()
        self.temperature = temperature
        self.w = w
    
    def forward(self, image_views: torch.Tensor, negative_views: torch.Tensor, other_embeddings: torch.Tensor) -> torch.Tensor:
        '''
        :param image_views: is a (2 x embedding dimension) tensor. first row corresponds to the first view while second row correspond to second view
        :param negative_views: is a (2 x n_negarives, embedding dimension tensor). We have stacked the embeddings of the first and the second view
        :param other_embeddings:
        '''
        inner_prod = image_views[0, :] @ image_views[1, :]
        self_similarity = torch.exp(torch.full((2, 1), fill_value=inner_prod.item(), device=str(image_views.device))/self.temperature)
        negatives_similarity = torch.exp(self.w * image_views @ negative_views.T/self.temperature)
        other_similarity = torch.exp(image_views @ other_embeddings.T/self.temperature)

        denominator = torch.sum(other_similarity, dim=1, keepdim=True) + torch.sum(negatives_similarity, dim=1, keepdim=True)
        return -torch.sum((self_similarity/denominator).log())


class SoftnnLossNegEuclidean(nn.Module):
    def __init__(self, temperature: float, w: float=2) -> None:
        super(SoftnnLossNegEuclidean, self).__init__()
        self.temperature = temperature
        self.w = w

    def forward(self, image_views: torch.Tensor, negative_views: torch.Tensor, other_embeddings: torch.Tensor) -> torch.Tensor:
        image_views_dist = torch.cdist(image_views[0, :].unsqueeze(0), image_views[1, :].unsqueeze(0))
        self_distance = torch.exp(torch.full((2, 1), fill_value=-image_views_dist.item(), device=str(image_views.device))/self.temperature)
        negatives_similarity = torch.exp(-self.w * torch.cdist(image_views, negative_views)/self.temperature)
        #print(negatives_similarity.shape)
        other_similarity = torch.exp(-torch.cdist(image_views, other_embeddings)/self.temperature)
        denominator = torch.sum(other_similarity, dim=1, keepdim=True) + torch.sum(negatives_similarity, dim=1, keepdim=True)
        return -torch.sum((self_distance/denominator).log())

#
# loss = SoftnnLossNegEuclidean(temperature=0.5)
# image_views = torch.randn((2, 100))
# negative_views= torch.randn((10, 100))
# other_embeddings = torch.randn((100, 100))
# print(loss(image_views, negative_views, other_embeddings))
#
# loss = InfoNCELossEuclidean(temperature=0.5)
# z_i = torch.randn(10, 100)
# z_j = torch.randn(10, 100)
# print(loss(z_i, z_j))


class SoftNNLossPos(nn.Module):
    def __init__(self, temperature: float, w: float = 2) -> None:
        super(SoftNNLossPos, self).__init__()
        self.temperature = temperature
        self.w = w

    def forward(self, image_views, positive_views, other_embeddings):
        positives_similarity = torch.exp(self.w * image_views @ positive_views.T/self.temperature)
        positives_similarity = torch.sum(positives_similarity, dim=1, keepdim=True)
        #print(positives_similarity.shape)
        other_similarity = torch.exp(image_views @ other_embeddings.T/self.temperature)
        #print(other_similarity.shape)

        denominator = torch.sum(other_similarity, dim=1, keepdim=True)
        return -torch.sum((positives_similarity/denominator).log())

class SoftNNLossPosEuclidean(nn.Module):
    def __init__(self, temperature: float, w: float = 2) -> None:
        super(SoftNNLossPosEuclidean, self).__init__()
        self.temperature = temperature
        self.w = w

    def forward(self, image_views, positive_views, other_embeddings):
        positives_similarity = torch.exp(-self.w * torch.cdist(image_views, positive_views)/self.temperature)
        positives_similarity = torch.sum(positives_similarity, dim=1, keepdim=True)

        other_similarity = torch.exp(-torch.cdist(image_views, other_embeddings)/self.temperature)
        denominator = torch.sum(other_similarity, dim=1, keepdim=True)
        return -torch.sum((positives_similarity/denominator).log())

# image_views = F.normalize(torch.randn(2, 128), dim=1)
# negative_views = F.normalize(torch.randn(5, 128), dim=1)
# other_views = F.normalize(torch.randn(100, 128), dim=1)
# loss = SoftNNLossPos(temperature=0.5)
# print(loss(image_views, negative_views, other_views))

class SoftNNLoss(nn.Module):
    def __init__(self, temperature: float, w: float = 2) -> None:
        super(SoftNNLoss, self).__init__()
        self.temperature = temperature
        self.w = w

    def forward(self, image_views, positive_views, negative_views, other_embeddings):
        positive_similarity = torch.exp(self.w * image_views @ positive_views.T/self.temperature)
        positive_similarity = torch.sum(positive_similarity, dim=1, keepdim=True)
        negatives_similarity = torch.exp(self.w * image_views @ negative_views.T/self.temperature)
        other_similarity = torch.exp(image_views @ other_embeddings.T/self.temperature)

        denominator = torch.sum(other_similarity, dim=1, keepdim=True) + torch.sum(negatives_similarity, dim=1, keepdim=True)
        return -torch.sum((positive_similarity/denominator).log())

class SoftNNLossEuclidean(nn.Module):
    def __init__(self, temperature: float, w: float = 2) -> None:
        super(SoftNNLossEuclidean, self).__init__()
        self.temperature = temperature
        self.w = w
    
    def forward(self, image_views, positive_views, negative_views, other_embeddings):
        positive_similarity = torch.exp(-self.w * torch.cdist(image_views, positive_views)/self.temperature)
        positive_similarity = torch.sum(positive_similarity, dim=1, keepdim=True)
        negatives_similarity = torch.exp(-self.w * torch.cdist(image_views, negative_views)/self.temperature)
        other_similarity = torch.exp(-torch.cdist(image_views, other_embeddings)/self.temperature)

        denominator = torch.sum(other_similarity, dim=1, keepdim=True) + torch.sum(negatives_similarity, dim=1, keepdim=True)
        return -torch.sum((positive_similarity/denominator).log())


def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'
    plt.switch_backend('Agg')
    X_embedded = TSNE(n_components=2).fit_transform(resnet_embeddings)
    unique_labels = np.unique(labels)

    num_colors = len(unique_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    for i, color in zip(unique_labels, colors):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=[color], label=f'Label {i}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Scatter Plot of Samples by Label')
    #plt.show()
    plt.savefig('NeuralNets/plots/first_plot.png')
    plt.switch_backend('TkAgg')



def find_indices_of_closest_embeddings(embedings: torch.Tensor, n_neighbors: int = 20, distance: str = 'cosine') -> torch.Tensor:
    assert (distance == 'cosine') | (distance == 'euclidean'), 'the distance must be cosine or euclidean'

    if (distance == 'cosine'):
        D = torch.matmul(embedings, embedings.T)
        indices = torch.topk(D, k=n_neighbors, dim=1)[1]
    elif (distance == 'euclidean'):
        D = torch.cdist(embedings, embedings)
        indices = torch.topk(D, k=n_neighbors, dim=1, largest=False)[1]
    return indices

def randomly_permute_tensor(x: torch.Tensor):
    n = x.size(0)
    rand_indices = torch.randperm(n)
    return x[rand_indices]


def MDS_closest_indices(embedding: torch.Tensor, n_neighbors: int = 20) -> torch.Tensor:
    embeddings_np = embedding.numpy()
    


def deterministic_closest_indices(Ids: torch.Tensor, n_neighbors: int = 20, n_correct: int = 16) -> torch.Tensor:
    indices = []
    n_samples = Ids.shape[0]
    n_false = n_neighbors - n_correct
    Ids_np = Ids.numpy()
    unique_Ids = np.unique(Ids_np)
    dictionary = {}
    for i in range(unique_Ids.shape[0]):
        x = unique_Ids[i]
        dictionary[x] = np.where(Ids_np == i)[0]
    #print(dictionary[0][0:5])
    
    for i in range(0, n_samples):
        current_id = Ids_np[i]
        correct_indices = np.random.choice(dictionary[current_id], size=n_correct, replace=False)
        new_id = np.random.choice(range(unique_Ids.shape[0]))
        while new_id == current_id:
            new_id = np.random.choice(range(unique_Ids.shape[0]))
        #print('new id is ', new_id)
        false_indices = np.random.choice(dictionary[new_id], size=n_false, replace=False)
        ii = np.hstack((correct_indices, false_indices))
        indices.append(ii)
    indices = np.vstack(indices)
    return torch.from_numpy(indices)



def probabilistic_closest_indices(Ids: torch.Tensor, n_neighbors: int = 20, n_correct_mean: int = 16) -> torch.Tensor:
    indices = []
    n_samples = Ids.shape[0]
    Ids_np = Ids.numpy()
    unique_Ids = np.unique(Ids_np)
    dictionary = {}
    for i in range(unique_Ids.shape[0]):
        x = unique_Ids[i]
        dictionary[x] = np.where(Ids_np == i)[0]
    
    for i in range(0, n_samples):
        current_id = Ids_np[i]
        n_correct = int(np.clip(np.random.normal(n_correct_mean, 5, 1), a_min=0, a_max=n_neighbors))

        n_false = n_neighbors - n_correct

        correct_indices = np.random.choice(dictionary[current_id], size=n_correct, replace=False)
        new_id = np.random.choice(range(unique_Ids.shape[0]))
        while new_id == current_id:
            new_id = np.random.choice(range(unique_Ids.shape[0]))
        
        false_indices = np.random.choice(dictionary[new_id], size=n_false, replace=False)
        ii = np.hstack((correct_indices, false_indices))
        indices.append(ii)
    
    indices = np.vstack(indices)
    return torch.from_numpy(indices)








# random_tensor = torch.randint(0, 10, size=(10000,))
# indices = probabilistic_closest_indices(random_tensor)



