import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *
from dataset import *
from augmentations import *


device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
# 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray, path2save: str, n_cpus: int=2) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'


    X_embedded = TSNE(n_components=2,perplexity=50, n_jobs=n_cpus).fit_transform(resnet_embeddings)
    unique_labels = np.unique(labels)

    num_colors = len(unique_labels)
    #colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for i, color in zip(unique_labels, colors):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=[color], label=f'Label {i}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Scatter Plot of Samples by Label')
    plt.savefig(path2save)
    #plt.show()


def VisualizeResNetEmbeddings(net, dataset_name: str='cifar100'):
    assert (dataset_name == 'cifar100'), 'i have currently implemented this function only for the CIFAR100 dataset' 
    dataset = CIFAR100()
    dataloader = DataLoader(dataset, batch_size=300, shuffle=False)
    embeddings = []
    Ids = []
    with torch.no_grad():
        for i, (X_batch, labels_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = net.forward(X_batch, forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())
            Ids.append(labels_batch)
        embeddings = torch.cat(embeddings, dim=0)
        Ids = torch.cat(Ids, dim=0)
    e = input('which embeddings should i visualize? type simclr or scan: ')
    assert (e == 'simclr') | (e == 'scan'), 'you should type simclr or scan'
    t = input('what classes whould i visualize? type all or seen or zero_shot')
    assert (t == 'all') | (t == 'seen') | (t == 'zero_shot'), 'type all or seen or zero_shot'
    p = 'NeuralNets/plots/'
    if e == 'simclr':
        if t == 'all':
            path = p + e + '_' + t + '.png'
            VisualizeWithTSNE(resnet_embeddings=embeddings.numpy(), labels=Ids.numpy(), path2save=path)
        elif t == 'seen':
            indices = torch.where(Ids != -1)[0]
            path = p + e + '_' + t + '.png'
            VisualizeWithTSNE(resnet_embeddings=embeddings[indices,:].numpy(), labels=Ids[indices].numpy(), path2save=path)
        elif t == 'zero_shot':
            path = p + e + '_' + t + '.png'
            indices = torch.where(Ids == -1)[0]
            VisualizeWithTSNE(resnet_embeddings=embeddings[indices].numpy(), labels=Ids[indices].numpy(), path2save=path)
    elif e == 'scan':
        if t == 'all':
            path = p + e + '_' + t + '.png'
            VisualizeWithTSNE(resnet_embeddings=embeddings.numpy(), labels=Ids.numpy(), path2save=path)
        elif t == 'seen':
            indices = torch.where(Ids != -1)[0]
            path = p + e + '_' + t + '.png'
            VisualizeWithTSNE(resnet_embeddings=embeddings[indices,:].numpy(), labels=Ids[indices].numpy(), path2save=path)
        elif t == 'zero_shot':
            path = p + e + '_' + t + '.png'
            indices = torch.where(Ids == -1)[0]
            VisualizeWithTSNE(resnet_embeddings=embeddings[indices].numpy(), labels=Ids[indices].numpy(), path2save=path)


def find_indices_of_closest_embeddings(embedings: torch.Tensor, n_neighbors: int = 20) -> torch.Tensor:
    D = torch.matmul(embedings, embedings.T)
    indices = torch.topk(D, k=n_neighbors, dim=1)[1]
    return indices


def plot_histogram_backbone_NN():
    clusternet = initializeClusterModel(dataset_name='cifar10', n_heads=1)
    id_aug = Identity_Augmentation(dataset_name='cifar10')
    dataset = CIFAR100()
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    embeddings = []
    clusternet.to(device)
    with torch.no_grad():
        for i, (X_batch, _, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            X_batch = id_aug(X_batch)
            embeddings_batch = clusternet.forward(X_batch, forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        neighbor_indices = find_indices_of_closest_embeddings(embeddings)
        Ids = dataset.Ids
        correct = []
        for i in range(0, Ids.numel()):
            neighbor_Ids = Ids[neighbor_indices[i, :]]
            n_correct = torch.where(neighbor_Ids == Ids[i])[0].numel()
            correct.append(n_correct)
        
        plt.figure(figsize=(8, 6))  # Set the figure size (optional)
        plt.hist(correct, bins=20, color='skyblue', edgecolor='black')  
        plt.title('Histogram of 20 Distinct Values') 
        plt.xlabel('Values') 
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.5)
        plt.show()


plot_histogram_backbone_NN()
