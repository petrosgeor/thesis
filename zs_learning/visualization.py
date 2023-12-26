import torch
import numpy as np
from dataset import *
from utils import *
from augmentations import *
from torch.utils.data import DataLoader


device = 'cuda'

def plot_histogram_backbone_NN():
    clusternet = initialize_clustering_net(n_classes=50, nheads=1)
    id_aug = Identity_Augmentation()
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
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










