import torch
import numpy as np
from dataset import *
from utils import *
from augmentations import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from evaluation import *



device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


clusternet = ClusteringModel(backbone=resnet18(), nclusters=50)
clusternet.load_state_dict(torch.load('NeuralNets/scan_trained_model.pth'))
# dataset = AwA2dataset()
# id_aug = Identity_Augmentation()

# dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
# clusternet.to(device)
# predictions = []
# embeddings = []
# labels = []
# with torch.no_grad():
#     for (X_batch, _, labels_batch, _) in dataloader:
#         X_batch = X_batch.to(device)
#         X_batch = id_aug(X_batch)
        
#         logits = clusternet.forward(X_batch)[0]
#         probs = F.softmax(logits, dim=1)
#         batch_predictions = torch.argmax(probs, dim=1)

#         predictions.append(batch_predictions.cpu())
#         labels.append(labels_batch)


# predictions = torch.cat(predictions, dim=0)
# labels = torch.cat(labels, dim=0)
# nmi, ari, acc = cluster_metric(label=labels.numpy(), pred=predictions.numpy())


def plot_histogram_backbone_NN():
    #clusternet = initialize_clustering_net(n_classes=50, nheads=1)
    id_aug = Identity_Augmentation()
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
    embeddings = []
    Ids = []
    clusternet.to(device)
    with torch.no_grad():
        for i, (X_batch, _, labels_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            X_batch = id_aug(X_batch)
            embeddings_batch = clusternet.forward(X_batch, forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())
            Ids.append(labels_batch)

        embeddings = torch.cat(embeddings, dim=0)
        Ids = torch.cat(Ids, dim=0)
        neighbor_indices = find_indices_of_closest_embeddings(F.normalize(embeddings, dim=1))
        #Ids = dataset.Ids
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


#plot_histogram_backbone_NN()










