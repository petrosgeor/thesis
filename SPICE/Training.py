import torch
from torch.nn import functional as F
import torch.optim as optim
from dataset import CIFAR100, plot_image_from_tensor, SCANDATASET
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from utils import *
from models import *
import numpy as np
from augmentations import *
from losses import losses
from evaluate import *



device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
# 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

torch.manual_seed(42)


def create_scan_dataset(net, n_neighbors=20, dataset_name='cifar100'):
    dataset = CIFAR100()

    cifar_dataloader = DataLoader(dataset, batch_size=300, shuffle=False)
    id_aug = Identity_Augmentation()
    embeddings = []
    with torch.no_grad():
        for i, (X_batch, _, _) in enumerate(cifar_dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = net.forward(id_aug(X_batch), forward_pass='backbone')
            embeddings.append(embeddings_batch)
        embeddings = torch.cat(embeddings, dim=0)
        all_neighbor_indices = find_indices_of_closest_embeddings(embedings=embeddings, masked_Ids=torch.from_numpy(dataset.masked_Ids), n_neighbors=n_neighbors)
    scan_dataset = SCANDATASET(data=torch.from_numpy(dataset.data), Ids=torch.from_numpy(dataset.Ids), 
                               masked_Ids=torch.from_numpy(dataset.masked_Ids), all_neighbors_indices=all_neighbor_indices)
    
    scan_dataloader = DataLoader(scan_dataset, batch_size=500, shuffle=True, num_workers=2)
    return scan_dataloader




def train_clustering_network(num_epochs: int, n_neighbors: int, dataset_name='cifar100'):
    clusternet = initializeClusterModel(freeze_backbone=False, dataset_name=dataset_name)
    if dataset_name == 'cifar10':
        num_clusters = 10
    elif dataset_name == 'cifar100':
        num_clusters = 20
    clusternet.to(device)

    id_aug = Identity_Augmentation()
    aug_clr = SimCLRaugment()
    dataloader = create_scan_dataset(net=clusternet, n_neighbors=n_neighbors, dataset_name=dataset_name)
    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4), weight_decay=10**(-4))
    ConsistencyLoss = losses.ClusterConsistencyLoss(threshold = -0.5)
    kl_loss = losses.KLClusterDivergance(num_clusters=num_clusters)


    for epoch in range(0, num_epochs):
        for i, (images, neighbor_images, _, _) in enumerate(dataloader):
            images = images.to(device)
            neighbor_images = neighbor_images.to(device)

            images_id = id_aug(images)
            images_clr = aug_clr(images)
            neighbor_images_id = id_aug(neighbor_images)

            probs = clusternet.forward(images_id)[0]
            probs_clr = clusternet.forward(images_clr)[0]
            probs_neighbors_id = clusternet.forward(neighbor_images_id)[0]

            loss1 = ConsistencyLoss.forward(probs1=probs, probs2=probs_neighbors_id) + ConsistencyLoss.forward(probs, probs_clr)

            loss2 = kl_loss.forward(probs=probs)

            total_loss = loss1 + 10*loss2
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch%20) == 0:
                true_labels = []
            predictions = []
            true_labels_conf = []
            predictions_conf = []
            with torch.no_grad():
                for i, (images_batch, _, labels_batch, _) in enumerate(dataloader):
                    images_batch = id_aug(images_batch.to(device))
                    batch_probs = clusternet.forward(images_batch)[0]
                    indices_conf = torch.where(batch_probs >= 0.95)
                    
                    true_labels_conf.append(labels_batch[indices_conf[0].cpu()])
                    predictions_conf.append(indices_conf[1].cpu())

                    batch_predictions = torch.argmax(batch_probs, dim=1)
                    predictions.append(batch_predictions.cpu())
                    true_labels.append(labels_batch)
                
                true_labels_conf = torch.cat(true_labels_conf, dim=0)
                predictions_conf = torch.cat(predictions_conf, dim=0)
                true_labels = torch.cat(true_labels, dim=0)
                predictions = torch.cat(predictions, dim=0)
                nmi, ari, acc = cluster_metric(label=true_labels.numpy(), pred=predictions.numpy())
                print('------------------- Epoch: ', epoch,' ---------------------')
                # Print the evaluation metrics
                print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
                print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
                print(f"Accuracy (ACC): {acc:.2f}%")
                print('\n')


train_clustering_network(num_epochs=50, n_neighbors=20, dataset_name='cifar100')














