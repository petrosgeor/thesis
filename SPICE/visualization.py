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


def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray, path2save: str) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'


    X_embedded = TSNE(n_components=2).fit_transform(resnet_embeddings)
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



net = initializeClusterModel(dataset_name='cifar100')
net.to(device)
dataset = CIFAR100()
id_aug = Identity_Augmentation()
dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
embeddings = []
Ids = []
with torch.no_grad():
    for i, (X_batch, labels_batch, _) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        embeddings_batch = net.forward(id_aug(X_batch), forward_pass='backbone')
        embeddings.append(embeddings_batch.cpu())
        Ids.append(labels_batch)
embeddings = torch.cat(embeddings, dim=0)
Ids = torch.cat(Ids, dim=0)
known_indices = torch.where(Ids != -1)[0]
zs_indices = torch.where(Ids == -1)[0]


