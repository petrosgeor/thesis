import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from models import *
from augmentations import *
from dataset import *
import numpy as np
#from kmeans_pytorch import kmeans
from clustering import *
import os
device = 'cuda'


# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def give_random_images(image_embeddings, number):
    pass


def contrastive_training(unsup_dataloader, sup_dataloader, num_epochs=2, t_contrastive=0.5):
    resnet, hidden_dim = get_resnet('resnet18')
    net = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    
    n_samples = unsup_dataloader.batch_size         # the batch size of the unsupervised dataloader (used for InfoNCE)
    net.float()
    net.to(device)

    augmentation = SimCLRaugment()
    id_augmentation = Identity_Augmentation()

    InfoNCE = InfoNCELossEuclidean(temperature=0.5)
    SoftPosLoss = SoftNNLossPosEuclidean(temperature=0.5)
    SoftNegLoss = SoftnnLossNegEuclidean(temperature=0.5)
    SoftLoss = SoftNNLossEuclidean(temperature=0.5)
    optimizer = optim.Adam(net.parameters(), lr=10**(-3))
    for epoch in range(num_epochs):
        #with torch.no_grad():
        if sup_dataloader is not None:
            dataloader_iterator = iter(sup_dataloader)
        for i, (X, _) in enumerate(unsup_dataloader):
            if sup_dataloader is not None:
                try:
                    image_batch, related_images_batch, relations_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(sup_dataloader)
                    image_batch, related_images_batch, relations_batch = next(dataloader_iterator)
            #print(i)
            X = X.to(device)
            z_i = net(augmentation(X))
            z_j = net(augmentation(X))
            loss1 = InfoNCE(z_i, z_j)

            loss2 = 0
            if sup_dataloader is not None:
                for j in range(0, image_batch.shape[0]):
                    indices_pos = torch.where(relations_batch[j, :] == 1)[0]
                    indices_neg = torch.where(relations_batch[j, :] == -1)[0]
                    image = image_batch[j, :].to(device)
                    image_views = torch.cat([augmentation(image).unsqueeze(0), augmentation(image).unsqueeze(0)], dim=0) # # shape is (2,3,32,32) for the two views
                    image_views_embeddings = net(image_views)
                    if indices_pos.numel() == 0:
                        image_neg = (related_images_batch[j, indices_neg]).to(device)
                        image_neg = torch.cat([augmentation(image_neg), augmentation(image_neg)], dim=0)    # shape is (2xn_neg, 3 ,32 ,32)
                        image_neg_embeddings = net(image_neg)
                        other_embeddings = torch.cat([z_i, z_j], dim=0)[torch.randint(0, 2*n_samples, size=(2*n_samples - image_neg.shape[0],)), :]
                        loss2 += SoftNegLoss(image_views=image_views_embeddings,
                                            negative_views=image_neg_embeddings,
                                            other_embeddings=other_embeddings)

                    elif indices_neg.numel() == 0:
                        image_pos = (related_images_batch[j, indices_pos]).to(device)
                        image_pos = torch.cat([augmentation(image_pos), augmentation(image_pos)], dim=0)
                        image_pos_embeddings = net(image_pos)
                        other_embeddings = torch.cat([z_i, z_j], dim=0)[torch.randint(0, 2*n_samples, size=(2*n_samples - image_pos.shape[0],)), :]
                        loss2 += SoftPosLoss(image_views=image_views_embeddings,
                                            positive_views=image_pos_embeddings,
                                            other_embeddings=other_embeddings)

                    else:
                        image_neg = (related_images_batch[j, indices_neg]).to(device)
                        image_neg = torch.cat([augmentation(image_neg), augmentation(image_neg)], dim=0)
                        image_neg_embeddings = net(image_neg)
                        image_pos = (related_images_batch[j, indices_pos]).to(device)
                        image_pos = torch.cat([augmentation(image_pos), augmentation(image_pos)], dim=0)
                        image_pos_embeddings = net(image_pos)
                        other_embeddings = torch.cat([z_i, z_j], dim=0)[torch.randint(0, 2*n_samples, size=(2*n_samples - image_neg.shape[0],)), :]
                        loss2 += SoftLoss(image_views=image_views_embeddings,
                                            positive_views=image_pos_embeddings,
                                            negative_views=image_neg_embeddings,
                                            other_embeddings=other_embeddings)
            
            total_loss = loss1 + loss2/(2*n_samples)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1)%5 == 0:
            with torch.no_grad():
                embeddings = []
                all_labels = []
                for j, (X, labels) in enumerate(unsup_dataloader):
                    X = X.to(device)
                    batch_embeddings = net.forward_r(id_augmentation(X))
                    embeddings.append(batch_embeddings)             # maybe i should move them to the cpu
                    all_labels.append(labels)

                embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                VisualizeWithTSNE(embeddings.cpu().numpy(), all_labels.numpy())
                #cluster_ids_x, cluster_centers = kmeans(X=embeddings, num_clusters=10,distance='euclidean', device=device)
                #print('for epoch: ', epoch, ' the NMI is: ', calculate_NMI(predictions=cluster_ids_x.cpu().numpy(), true_labels=all_labels.numpy()))
            train_cluster_head(embeddings.cpu().numpy(), all_labels.numpy(), n_neighbors=20)


dataset = CIFAR10()
linked_dataset = LinkedDataset(dataset, num_links=200)

dataloader1 = DataLoader(dataset, batch_size=2000)
dataloader2 = DataLoader(linked_dataset, batch_size=100)
dataloader2 = None

contrastive_training(dataloader1, dataloader2, num_epochs=100)


