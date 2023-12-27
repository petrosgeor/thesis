from utils import *
from evaluation import *
from augmentations import *
from dataset import *
from models import *
from losses import clusterlosses

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def scan_training(num_epochs: int=200, num_classes:int = 50):
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    clusternet = ClusteringModel(backbone=resnet18(), nclusters=num_classes, nheads=1)
    clusternet.to(device)

    id_aug = Identity_Augmentation()
    aug_clr = SimCLRaugment()
    ScanLoss = clusterlosses.SCANLoss()
    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4), weight_decay=10**(-4))
    earlystopping = EarlyStopping(patience=3, delta=0.01)
    for epoch in range(0, num_epochs):
        for i, (images_u, neighbor_images, _, _) in enumerate(dataloader):
            images_u = images_u.to(device)
            neighbor_images = neighbor_images.to(device)
            images_u_id = id_aug(images_u)  # identity augmentation
            images_u_clr = aug_clr(images_u)
            neighbor_images_id = id_aug(neighbor_images)


            anchors_id = clusternet.forward(images_u_id)[0]
            anchors_clr = clusternet.forward(images_u_clr)[0]
            neighbors_id = clusternet.forward(neighbor_images_id)[0]
            #total_loss, _, _ = ScanLoss.forward(anchors=anchors_id, neighbors=neighbors_id) + ScanLoss.forward(anchors=anchors_id, neighbors=anchors_clr) #+ ConsistencyLoss.forward(probs1=probs, probs2=probs_neighbors_clr, weights=weights) 
            total_loss, _, _ = ScanLoss.forward(anchors=torch.cat([anchors_id, anchors_id]), neighbors=torch.cat([neighbors_id, anchors_clr]))

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.save(clusternet.backbone.state_dict(), 'NeuralNets/backbone_AwA2.pth')
        torch.save(clusternet.cluster_head.state_dict(), 'NeuralNets/cluster_head_AwA2.pth')

        
        if (epoch%5) == 0:
            true_labels = []
            predictions = []
            true_labels_conf = []
            predictions_conf = []
            with torch.no_grad():
                for i, (images_batch, _, labels_batch, _) in enumerate(dataloader):
                    images_batch = id_aug(images_batch.to(device))
                    batch_probs = F.softmax(clusternet.forward(images_batch)[0], dim=1)
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
            earlystopping(val_accuracy=acc)
            if earlystopping.early_stop == True:
                break
            

scan_training(num_epochs=200, num_classes=50)