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
run_pretraining = input("do you want to run the pretraining step? ")
assert (run_pretraining == 'yes') | (run_pretraining == 'no'), 'the answer must be yes or no'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def give_random_images(image_embeddings, number):
    pass


def contrastive_training(unsup_dataloader, sup_dataloader, num_epochs=2, t_contrastive=0.5, consider_links: bool = False):
    resnet, hidden_dim = get_resnet('resnet34')
    net = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    
    n_samples = unsup_dataloader.batch_size         # the batch size of the unsupervised dataloader (used for InfoNCE)
    net.float()
    net.to(device)

    augmentation = SimCLRaugment()
    id_augmentation = Identity_Augmentation()

    InfoNCE = InfoNCELoss(temperature=0.5)
    SoftPosLoss = SoftNNLossPos(temperature=0.5)
    SoftNegLoss = SoftNNLossNeg(temperature=0.5)
    SoftLoss = SoftNNLoss(temperature=0.5)
    optimizer = optim.Adam(net.parameters(), lr=10**(-3))
    for epoch in range(num_epochs):
        #with torch.no_grad():
        if consider_links == True:
            dataloader_iterator = iter(sup_dataloader)
        for i, (X, _) in enumerate(unsup_dataloader):
            if consider_links == True:
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
            if consider_links == True:
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
                    batch_embeddings = net(id_augmentation(X))
                    embeddings.append(batch_embeddings)             # maybe i should move them to the cpu
                    all_labels.append(labels)

                embeddings = (torch.cat(embeddings, dim=0)).cpu()
                all_labels = torch.cat(all_labels, dim=0)
                #VisualizeWithTSNE(embeddings.cpu().numpy(), all_labels.numpy())
                #cluster_ids_x, cluster_centers = kmeans(X=embeddings, num_clusters=10,distance='euclidean', device=device)
                #print('for epoch: ', epoch, ' the NMI is: ', calculate_NMI(predictions=cluster_ids_x.cpu().numpy(), true_labels=all_labels.numpy()))
            #train_cluster_head(embeddings, all_labels, n_neighbors=20)
            print('For epoch ', epoch, 'the NMI is: ', train_cluster_head(embeddings, all_labels, n_neighbors=20))
    if consider_links == False:
        torch.save(net.state_dict(), 'NeuralNets/ResNetBackbone.pth')
    elif consider_links == True:
        torch.save(net.state_dict(), 'NeuralNets/ResNetBackboneLinks')
    return net
        
def VisualizedResNetBackBoneEmbeddings():
    resnet, hidden_dim = get_resnet('resnet34')
    net = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    net.load_state_dict(torch.load('NeuralNets/ResNetBackbone.pth'))
    dataset = CIFAR10()
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    embeddings = []
    labels = []
    net.float()
    net.to(device)
    aug = Identity_Augmentation()
    with torch.no_grad():
        for i, (X_batch, labels_batch) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            X_batch = aug(X_batch)
            batch_embeddings = net.forward_r(X_batch)
            
            embeddings.append(batch_embeddings.cpu())
            labels.append(labels_batch)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

    VisualizeWithTSNE(resnet_embeddings=embeddings.numpy(), labels=labels.numpy())


def create_SCAN_dl_LINKED_dl(net: Network, pretrained: str) -> tuple:
    dataset = CIFAR10()
    linked_dataset = LinkedDataset(dataset, num_links=5000)
    cifar_dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    id_aug = Identity_Augmentation()
    embeddings = []
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(cifar_dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = net(id_aug(X_batch))
            embeddings.append(embeddings_batch.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        neighbor_indices = find_indices_of_closest_embeddings(embeddings, distance='cosine')
    scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
    

def train_clustering_network(num_epochs=2, t_contrastive=0.5, consider_links: bool = False):
    pretrained = input('which PRETRAINED model should i consider, the one with links or without? type links or no_links')
    assert (pretrained == 'links') | (pretrained == 'no_links'), 'please type links or no_links'
    resnet, hidden_dim = get_resnet('resnet34')
    clusternet = Network(resnet=resnet, hidden_dim=hidden_dim)
    if pretrained == 'no_links':
        clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackbone.pth'))
    elif pretrained == 'links':
        clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackboneLinks'))
    
    clusternet.to(device)
    dataset = CIFAR10()
    linked_dataset = LinkedDataset(dataset, num_links=5000)
    cifar_dataloader = DataLoader(CIFAR10, batch_size=500, shuffle=False)
    id_aug = Identity_Augmentation()
    embeddings = []
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(cifar_dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = clusternet(id_aug(X_batch))
            embeddings.append(embeddings_batch.cpu())
        
    embeddings = torch.cat(embeddings, dim=0)
    neighbor_indices = find_indices_of_closest_embeddings(embeddings, distance='cosine')

    ############    SCAN DATASET    #######
    scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
    scan_dataloader = DataLoader(scan_dataset, batch_size=128, shuffle=True)
    links_dataloader = DataLoader(linked_dataset, batch_size=128, shuffle=True)

    for epoch in range(0, num_epochs):
        if consider_links == True:
            dataloader_iterator = iter(links_dataloader)
        for i, (images_u, _, neighbor_images) in enumerate(scan_dataloader):
            if consider_links == True:
                try:
                    image_l, related_images, relations_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(links_dataloader)
                    image_l, related_images, relations_batch = next(dataloader_iterator)
            
            ####    SCAN LOSS   ####
            images_u = id_aug(images_u.to(device))
            neighbor_images = neighbor_images.to(device)
            neighbor_images = neighbor_images.reshape()
            embeddings = clusternet.forward_c(images_u)
            neighbor_embeddings = clusternet.forward_c()
            








def run_pretraining_function():
    if run_pretraining == 'yes':
        consider_links = input('do you want to consider any links?')
        assert (consider_links == 'yes') | (consider_links == 'no'), 'the answer must be yes or no'
        dataset = CIFAR10()
        linked_dataset = LinkedDataset(dataset, num_links=5000)

        dataloader1 = DataLoader(dataset, batch_size=1500, shuffle=True)
        dataloader2 = DataLoader(linked_dataset, batch_size=100)
        if consider_links == 'no':
            net = contrastive_training(dataloader1, dataloader2, num_epochs=300, consider_links=False)
        elif consider_links == 'yes':
            net = contrastive_training(dataloader1, dataloader2, num_epochs=300, consider_links=True)
    else:
        return 'no pretraining will take place'

run_pretraining_function()