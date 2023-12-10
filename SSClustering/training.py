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
from evaluate import *
import os
from losses.losses import *
device = 'cuda'


# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
run_pretraining = input("do you want to run the pretraining step? ")
assert (run_pretraining == 'yes') | (run_pretraining == 'no'), 'the answer must be yes or no'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def give_random_images(image_embeddings, number):
    pass


def contrastive_training(unsup_dataloader, sup_dataloader, num_epochs=2, t_contrastive=0.5, consider_links: bool = False):
    resnet, hidden_dim = get_resnet('resnet18')
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
    
        if (epoch + 1)%10 == 0:
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


def create_SCAN_dl_LINKED_dl(net: Network) -> tuple:   # creates dataloaders for both the SCAN and LINKED datasets
    dataset = CIFAR10()
    linked_dataset = LinkedDataset(dataset, num_links=5000)
    cifar_dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    id_aug = Identity_Augmentation()
    embeddings = []
    net.eval()
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(cifar_dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = net(id_aug(X_batch))
            embeddings.append(embeddings_batch.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        neighbor_indices = find_indices_of_closest_embeddings(embeddings, distance='cosine', n_neighbors=20)
    scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
    scan_dataloader = DataLoader(scan_dataset, batch_size=128, shuffle=True)
    linked_dataloader = DataLoader(linked_dataset, batch_size=128, shuffle=True)
    return scan_dataloader, linked_dataloader


def train_clustering_network(num_epochs=2, t_contrastive=0.5, consider_links: bool = False):
    pretrained = input('which PRETRAINED model should i consider, the one with links or without? type links or no_links: ')
    assert (pretrained == 'links') | (pretrained == 'no_links'), 'please type links or no_links'
    resnet, hidden_dim = get_resnet('resnet34')
    clusternet = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    if pretrained == 'no_links':
        clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackbone.pth'))
    elif pretrained == 'links':
        clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackboneLinks'))
    
    clusternet.to(device)
    id_aug = Identity_Augmentation()
    
    scan_dataloader, linked_dataloader = create_SCAN_dl_LINKED_dl(net=clusternet)
    n_neighbors = scan_dataloader.dataset.n_neighbors
    n_classes = (torch.unique(scan_dataloader.dataset.Ids)).numel()
    ####
    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4))
    ConsistencyLoss = ClusterConsistencyLoss()
    EntropyLoss = ClusterEntropyLoss()

    print('the mean of images with same neighbors is: ', np.mean(scan_dataloader.dataset.same_Ids_list))
    clusternet.train()
    for epoch in range(0, num_epochs):
        if consider_links == True:
            dataloader_iterator = iter(linked_dataloader)
        for i, (images_u, _, neighbor_images) in enumerate(scan_dataloader):
            if consider_links == True:
                try:
                    image_l, related_images, relations_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(linked_dataloader)
                    image_l, related_images, relations_batch = next(dataloader_iterator)
            n_images_u = images_u.shape[0]
            ####    SCAN LOSS   ####
            images_u = id_aug(images_u.to(device))
            neighbor_images = id_aug(neighbor_images.to(device))
            neighbor_images = neighbor_images.reshape(n_images_u * n_neighbors, 3, 32, 32)
            probs = clusternet.forward_c(images_u)
            probs_neighbors = clusternet.forward_c(neighbor_images)
            probs_neighbors = probs_neighbors.reshape(n_images_u, n_neighbors, n_classes)

            loss1 = ConsistencyLoss.forward(probs, probs_neighbors)
            loss2 = 0
            # if consider_links == True:
            #     print('if this print we have a problem')
            #     for j in range(0, image_l.shape[0]):
            #         indices_pos = torch.where(relations_batch[j, :] == 1)[0]
            #         indices_neg = torch.where(relations_batch[j, :] == -1)[0]
            #         image = image_l[j, :].to(device)
            #         image_probs = clusternet.forward_c(image)

            #         if indices_pos.numel() == 0:
            #             images_neg = (related_images[j, indices_neg]).to(device)
            #             image_neg_probs = clusternet.forward_c(images_neg)
            #             loss2 = loss2 + (torch.matmul(image_neg_probs, image_probs).log()).sum()
                    
            #         elif indices_neg.numel() == 0:
            #             images_pos = (related_images[j, indices_pos]).to(device)
            #             images_pos_probs = clusternet.forward_c(images_pos)
            #             loss2 = loss2 - (torch.matmul(images_pos_probs, image_probs).log()).sum()
                    
            #         else: 
            #             images_neg = (related_images[j, indices_neg]).to(device)
            #             image_neg_probs = clusternet.forward_c(images_neg)
            #             images_pos = (related_images[j, indices_pos]).to(device)
            #             images_pos_probs = clusternet.forward_c(images_pos)

            #             loss2 = loss2 + (torch.matmul(image_neg_probs, image_probs).log()).sum()
            #             loss2 = loss2 - (torch.matmul(images_pos_probs, image_probs).log()).sum()
            loss3 = EntropyLoss.forward(probs)

            total_loss = loss1 + loss2/(linked_dataloader.batch_size) + 5*loss3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            predictions = []
            true_labels = []
            for i, (images_u, labels_batch, _) in enumerate(scan_dataloader):
                images_u = (id_aug(images_u)).to(device)
                batch_probs = clusternet.forward_c(images_u)
                predictions.append(torch.argmax(batch_probs, dim=1).cpu())
                true_labels.append(labels_batch)
            
            predictions = torch.cat(predictions, dim=0)
            true_labels = torch.cat(true_labels, dim=0)
        #print('Epoch ',epoch,' NMI is: ', calculate_NMI(predictions=predictions.numpy(), true_labels=true_labels.numpy()))
        print('----------- Epoch ', epoch, ' -------------')
        nmi, ari, acc = cluster_metric(true_labels.numpy(), predictions.numpy())
        print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
        print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
        print(f"Accuracy (ACC): {acc:.2f}%")












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
#train_clustering_network(num_epochs=20, consider_links=False)


