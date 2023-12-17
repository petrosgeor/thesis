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
from losses import losses
from tests import test2
from visualization import pretrained_vis



device = 'cuda'


# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
# 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def contrastive_training(unsup_dataloader, sup_dataloader, num_epochs=2, t_contrastive=0.5, consider_links: bool = False):
    resnet, hidden_dim = get_resnet('resnet18')
    net = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    
    n_samples = unsup_dataloader.batch_size         # the batch size of the unsupervised dataloader (used for InfoNCE)
    net.float()
    net.to(device)

    augmentation = SimCLRaugment()
    id_augmentation = Identity_Augmentation()

    InfoNCE = losses.InfoNCELoss(temperature=t_contrastive)
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
            z_i = net(id_augmentation(X))
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
        

def create_SCAN_dl_LINKED_dl(net: Network, take_neighbors = 'neuralnet', n_neighbors=20) -> tuple:   # creates dataloaders for both the SCAN and LINKED datasets
    dataset = CIFAR10(proportion=1)
    linked_dataset = LinkedDataset(dataset, num_links=5000)
    assert (take_neighbors == 'neuralnet') | (take_neighbors == 'neuralnet') | (take_neighbors == 'probabilistic') | (take_neighbors == 'paiper')
    if take_neighbors == 'neuralnet':
        cifar_dataloader = DataLoader(dataset, batch_size=500, shuffle=False)  
        id_aug = Identity_Augmentation()
        embeddings = []
        with torch.no_grad():
            for i, (X_batch, Ids) in enumerate(cifar_dataloader):
                X_batch = X_batch.to(device)
                embeddings_batch = net.forward_r(id_aug(X_batch))
                embeddings.append(embeddings_batch.cpu())
            
            embeddings = torch.cat(embeddings, dim=0)
            neighbor_indices = find_indices_of_closest_embeddings(embeddings, distance='cosine', n_neighbors=n_neighbors)
            scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)

    elif take_neighbors == 'paiper':
        id_aug = test2.val_augmentation()
        cifar_dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
        embeddings = []
        labels = []
        with torch.no_grad():
            for i, (X_batch, Ids) in enumerate(cifar_dataloader):
                X_batch = X_batch.to(device)
                embeddings_batch = net.forward(id_aug(X_batch), forward_pass='backbone')
                embeddings.append(embeddings_batch.cpu())
                labels.append(Ids)
            embeddings = torch.cat(embeddings, dim=0)
            labels = torch.cat(labels, dim=0)

            neighbor_indices = find_indices_of_closest_embeddings(embeddings, distance='cosine', n_neighbors=n_neighbors)
            scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
            #pretrained_vis.VisualizeWithTSNE(resnet_embeddings=embeddings.numpy(), labels=labels.numpy())
    elif take_neighbors == 'probabilistic':
        neighbor_indices = probabilistic_closest_indices(Ids=dataset.Ids, n_neighbors=n_neighbors, n_correct_mean=9)
        scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
    elif take_neighbors == 'deterministic':
        neighbor_indices = deterministic_closest_indices(Ids=dataset.Ids, n_neighbors=n_neighbors, n_correct=9)
        scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)

    #scan_dataset = SCANdatasetWithNeighbors(data=dataset.data, Ids=dataset.Ids, neighbor_indices=neighbor_indices)
    scan_dataloader = DataLoader(scan_dataset, batch_size=512, shuffle=True, num_workers=2)
    linked_dataloader = DataLoader(linked_dataset, batch_size=256, shuffle=True, num_workers=2)
    return scan_dataloader, linked_dataloader


def train_clustering_network(num_epochs=2, t_contrastive=0.5, consider_links: bool = False, n_neighbors=20, testing=False, take_neighbors='neuralnet'):
    #pretrained = input('which PRETRAINED model should i consider, the one with links or without? type links or no_links: ')
    pretrained = 'no_links'
    assert (take_neighbors == 'neuralnet') | (take_neighbors == 'probabilistic') | (take_neighbors == 'paiper'), 'take_neighbors must be neuralnet or probabilistic'
    assert (pretrained == 'links') | (pretrained == 'no_links'), 'please type links or no_links'
    if take_neighbors != 'paiper':
        resnet, hidden_dim = get_resnet('resnet18')
        clusternet = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=10)
    elif take_neighbors == 'paiper':
        backbone = test2.resnet18()
        contrastivemodel = test2.ContrastiveModel(backbone=backbone)
        file_path = 'NeuralNets/simclr_cifar10.pth'
        checkpoint = torch.load(file_path)
        contrastivemodel.load_state_dict(checkpoint)
        clusternet = test2.ClusteringModel(backbone={'backbone': contrastivemodel.backbone, 'dim': contrastivemodel.backbone_dim}, nclusters=10)

    if take_neighbors != 'paiper':
        if pretrained == 'no_links':
            clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackbone.pth'))
        elif pretrained == 'links':
            clusternet.load_state_dict(torch.load('NeuralNets/ResNetBackboneLinks'))
        
    clusternet.to(device)
    id_aug = Identity_Augmentation()
    aug_clr = SimCLRaugment()
    scan_dataloader, linked_dataloader = create_SCAN_dl_LINKED_dl(net=clusternet, take_neighbors=take_neighbors, n_neighbors=n_neighbors)
    if testing == True:
        return scan_dataloader
    optimizer = optim.SGD(clusternet.parameters(), lr=10**(-2))
    ConsistencyLoss = losses.ClusterConsistencyLoss()
    kl_loss = losses.KLClusterDivergance()

    print('the mean of images with same neighbors is: ', np.mean(scan_dataloader.dataset.same_Ids_list))
    print('the variance of images with same neighbors is ', np.var(scan_dataloader.dataset.same_Ids_list))
    print('the mean of neighborhood consistecy links is: ', np.mean(scan_dataloader.dataset.correct_links_list))
    clusternet.train()
    for epoch in range(0, num_epochs):
        if consider_links == True:
            dataloader_iterator = iter(linked_dataloader)
        for i, (images_u, _, neighbor_images) in enumerate(scan_dataloader):
            if consider_links == True:
                try:
                    image_l, related_images, relations = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(linked_dataloader)
                    image_l, related_images, relations = next(dataloader_iterator)

            ####    SCAN LOSS   ####
            loss1 = 0

            images_u = images_u.to(device)
            images_u_id = id_aug(images_u)  # identity augmentation
            images_u_clr = aug_clr(images_u)
            neighbor_images = id_aug(neighbor_images.to(device))
            probs = clusternet.forward_c(images_u_id)   # probabilites of identity images
            probs_clr = clusternet.forward_c(images_u_clr)  # probabilities of CLR images
            probs_neighbors = clusternet.forward_c(neighbor_images)
            loss1 = ConsistencyLoss.forward(probs1=probs, probs2=probs_neighbors, relations=None) + ConsistencyLoss.forward(probs1=probs, probs2=probs_clr)

            ####    LINKED IMAGES LOSS  ####
            loss2 = 0
            if consider_links == True:
                image_l = image_l.to(device)
                image_l_id = id_aug(image_l)
                image_l_clr = aug_clr(image_l)
                related_images = related_images.to(device)
                related_images_id = id_aug(related_images)
                related_images_clr = aug_clr(related_images)
                relations = relations.to(device)
                p_id = clusternet.forward_c(image_l_id)
                p_linked_id = clusternet.forward_c(related_images_id)
                p_clr = clusternet.forward_c(image_l_clr)
                p_linked_clr = clusternet.forward_c(related_images_clr)

                loss2 = ConsistencyLoss(probs1=p_id, probs2=p_linked_id, relations=relations) + ConsistencyLoss(p_clr, p_linked_clr, relations)
            #loss3 = EntropyLoss.forward(probs=probs)
            loss3 = 0
            loss3 = kl_loss.forward(probs=probs)
            total_loss = loss1 + 10**(-2)*loss2 + 10*loss3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1)%10 == 0:
            true_labels = []
            predictions = []
            true_labels_conf = []
            predictions_conf = []
            with torch.no_grad():
                for i, (images_batch, labels_batch, _) in enumerate(scan_dataloader):
                    images_batch = id_aug(images_batch.to(device))
                    batch_probs = clusternet.forward_c(images_batch)
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
                print('confident examples \n')
                nmi, ari, acc = cluster_metric(label=true_labels_conf.numpy(), pred=predictions_conf.numpy())
                print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
                print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
                print(f"Accuracy (ACC): {acc:.2f}%")

            x = torch.unique(predictions, return_inverse=False, return_counts=True)


def create_unified_dataset(net, n_neighbors=20, return_distances: bool = True, num_links: int=5000) -> tuple:
    dataset = CIFAR10(proportion=1)
    cifar_dataloader = DataLoader(dataset, batch_size=300, shuffle=False)
    id_aug = Identity_Augmentation()
    embeddings = []
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(cifar_dataloader):
            X_batch = X_batch.to(device)
            embeddings_batch = net.forward(id_aug(X_batch), forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        distances, indices = find_indices_of_closest_embeddings(embeddings, n_neighbors=n_neighbors, return_distances=return_distances)
        unified_dataset = UnifiedDataset(data=dataset.data, Ids=dataset.Ids, neighbor_indices=indices,
                                         neighbors_distances=distances, num_links=num_links)
        
        dataloader = DataLoader(dataset=unified_dataset, batch_size=500, shuffle=True)
        return dataloader


def train_clustering_network2(num_epochs=2, t_contrastive=0.5, consider_links: bool = False, n_neighbors=20, testing=False):
    clusternet = initializeClusterModel(freeze_backbone=False)
    clusternet.to(device)

    id_aug = Identity_Augmentation()
    aug_clr = SimCLRaugment()
    scan_dataloader, linked_dataloader = create_SCAN_dl_LINKED_dl(net=clusternet, take_neighbors='paiper', n_neighbors=n_neighbors)
    if testing == True:
        return scan_dataloader
    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4), weight_decay=10**(-4))
    ConsistencyLoss = losses.ClusterConsistencyLoss()
    kl_loss = losses.KLClusterDivergance()
    #clusternet.train()

    for epoch in range(0, num_epochs):
        if consider_links == True:
            dataloader_iterator = iter(linked_dataloader)
        for i, (images_u, _, neighbor_images) in enumerate(scan_dataloader):
            if consider_links == True:
                try:
                    image_l, related_images, relations = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(linked_dataloader)
                    image_l, related_images, relations = next(dataloader_iterator)

            ####    SCAN LOSS   ####
            loss1 = 0
            images_u = images_u.to(device)
            images_u_id = id_aug(images_u)  # identity augmentation
            images_u_clr = aug_clr(images_u)
            neighbor_images = id_aug(neighbor_images.to(device))
            probs = clusternet.forward(images_u_id)[0]
            probs_clr = clusternet.forward(images_u_clr)[0]
            probs_neighbors = clusternet.forward(neighbor_images)[0]
            loss1 = ConsistencyLoss.forward(probs1=probs, probs2=probs_neighbors, relations=None) + ConsistencyLoss.forward(probs1=probs, probs2=probs_clr)

            loss2 = 0
            if consider_links == True:
                image_l = image_l.to(device)
                image_l_id = id_aug(image_l)
                image_l_clr = aug_clr(image_l)
                related_images = related_images.to(device)
                related_images_id = id_aug(related_images)
                related_images_clr = aug_clr(related_images)
                relations = relations.to(device)
                p_id = clusternet.forward_c(image_l_id)
                p_linked_id = clusternet.forward_c(related_images_id)
                p_clr = clusternet.forward_c(image_l_clr)
                p_linked_clr = clusternet.forward_c(related_images_clr)

                loss2 = ConsistencyLoss(probs1=p_id, probs2=p_linked_id, relations=relations) + ConsistencyLoss(p_clr, p_linked_clr, relations)

            loss3 = 0
            loss3 = kl_loss.forward(probs=probs)
            total_loss = loss1 + 10**(-2)*loss2 + 10*loss3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1)%10 == 0:
            true_labels = []
            predictions = []
            true_labels_conf = []
            predictions_conf = []
            with torch.no_grad():
                for i, (images_batch, labels_batch, _) in enumerate(scan_dataloader):
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
                print('confident examples \n')
                nmi, ari, acc = cluster_metric(label=true_labels_conf.numpy(), pred=predictions_conf.numpy())
                print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
                print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
                print(f"Accuracy (ACC): {acc:.2f}%")


def train_clustering_network3(num_epochs:int=50, n_neighbors:int=20, consider_distnaces:bool=True, num_links:int=5000):
    clusternet = initializeClusterModel(freeze_backbone=False)
    clusternet.to(device)

    id_aug = Identity_Augmentation()
    aug_clr = SimCLRaugment()
    dataloader = create_unified_dataset(net=clusternet, n_neighbors=n_neighbors, return_distances=consider_distnaces, num_links=num_links)

    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4), weight_decay=10**(-4))
    ConsistencyLoss = losses.ClusterConsistencyLoss(threshold = -3)
    kl_loss = losses.KLClusterDivergance()

    for epoch in range(0, num_epochs):
        for i, (images_u, neighbor_images, weights, _) in enumerate(dataloader):
            images_u = images_u.to(device)
            images_u_id = id_aug(images_u)  # identity augmentation
            images_u_clr = aug_clr(images_u)
            neighbor_images = id_aug(neighbor_images.to(device))
            weights = weights.to(device)
            probs = clusternet.forward(images_u_id)[0]
            probs_clr = clusternet.forward(images_u_clr)[0]
            probs_neighbors = clusternet.forward(neighbor_images)[0]
            loss1 = ConsistencyLoss.forward(probs1=probs, probs2=probs_neighbors, weights=weights) + ConsistencyLoss.forward(probs1=probs, probs2=probs_clr, weights=None)

            loss2 = kl_loss.forward(probs=probs)

            total_loss = loss1 + 10*loss2
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1)%5 == 0:
            true_labels = []
            predictions = []
            true_labels_conf = []
            predictions_conf = []
            with torch.no_grad():
                for i, (images_batch, _, _, labels_batch) in enumerate(dataloader):
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
                print('confident examples \n')
                # nmi, ari, acc = cluster_metric(label=true_labels_conf.numpy(), pred=predictions_conf.numpy())
                # print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
                # print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
                # print(f"Accuracy (ACC): {acc:.2f}%")


def run_pretraining_function():
    run_pretraining = input("do you want to run the pretraining step? ")
    assert (run_pretraining == 'yes') | (run_pretraining == 'no'), 'the answer must be yes or no'
    if run_pretraining == 'yes':
        consider_links = input('do you want to consider any links?')
        assert (consider_links == 'yes') | (consider_links == 'no'), 'the answer must be yes or no'
        dataset = CIFAR10()
        linked_dataset = LinkedDataset(dataset, num_links=5000)

        dataloader1 = DataLoader(dataset, batch_size=512, shuffle=True)
        dataloader2 = DataLoader(linked_dataset, batch_size=100)
        if consider_links == 'no':
            net = contrastive_training(dataloader1, dataloader2, num_epochs=500, consider_links=False, t_contrastive=0.7)
        elif consider_links == 'yes':
            net = contrastive_training(dataloader1, dataloader2, num_epochs=500, consider_links=True, t_contrastive=0.7)
    else:
        return 'no pretraining will take place'

train_clustering_network3(num_epochs=101, n_neighbors=20, consider_distnaces=False, num_links=20000)


# scan_dataloader = train_clustering_network(num_epochs=300, t_contrastive=0.5, consider_links = True, n_neighbors=20,
#                                            testing=True, take_neighbors='paiper')
# Ids = scan_dataloader.dataset.Ids
# neighbors = scan_dataloader.dataset.neighbor_indices
# class_correct = []
# for i in range(0, 10):
#     current_indices = torch.where(Ids == i)[0]
#     i_class_current = []
#     for j in current_indices:
#         current_neighbor_indices = neighbors[j.item(), :]
#         i_class_current.append(torch.where(Ids[current_neighbor_indices] == i)[0].numel())
#     class_correct.append(np.mean(i_class_current))

# print(class_correct)

# same_Ids_list = scan_dataloader.dataset.same_Ids_list
# plt.figure(figsize=(8, 6))  # Set the figure size (optional)
# plt.hist(same_Ids_list, bins=20, color='skyblue', edgecolor='black')  # Create a histogram with 20 bins
# plt.title('Histogram of 20 Distinct Values')  # Set the title of the histogram
# plt.xlabel('Values')  # Set the label for the x-axis
# plt.ylabel('Frequency')  # Set the label for the y-axis
# plt.grid(axis='y', alpha=0.5)  # Add gridlines for better visualization

# plt.savefig('NeuralNets/plots/histogram.png')
#plt.show()



#a = scan_dataloader.dataset.correct_links_list
# print(np.mean(a))
# neighbor_indices = scan_dataloader.dataset.neighbor_indices
# Ids = scan_dataloader.dataset.Ids
# correct = []
# for i in range(0, Ids.shape[0]):
#     current_id = Ids[i]
#     x = torch.where(Ids[neighbor_indices[i,:]] == current_id)[0].numel()
#     correct.append(x)













