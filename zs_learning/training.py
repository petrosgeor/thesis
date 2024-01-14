from utils import *
from evaluation import *
from augmentations import *
from dataset import *
from models import *
from losses import clusterlosses

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id



dataset = AwA2dataset(training=True, size_resize=224)
def scan_training(num_epochs: int=200, num_classes:int = 50):
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    dataloader = DataLoader(dataset, batch_size=190, shuffle=True, num_workers=2)

    clusternet = ClusteringModel(backbone={'backbone':resnet, 'dim':512}, nclusters=num_classes, nheads=1)
    clusternet.to(device)

    scanloss = clusterlosses.SCANLoss(entropy_weight=1.0)

    zs_Ids = dataset.zs_Ids.to(device)
    known_Ids = dataset.known_Ids.to(device)

    num_zs_Ids = zs_Ids.numel()
    K = 30
    dummy = zs_Ids.repeat(K, 1).T
    optimizer = optim.Adam(clusternet.parameters(), lr=10**(-4), weight_decay=10**(-4))
    earlystopping = EarlyStopping(patience=6, delta=0.01)
    num_gradient_steps = 7


    for epoch in range(0, num_epochs):
        clusternet.train()
        epoch_loss = 0
        for i, (images, _, neighbors, Ids, _) in enumerate(dataloader):
            Ids = Ids.to(device)
            images = images.to(device)
            neighbors = neighbors.to(device)

            # the logits for each image and their corresponding neighbors
            anchors_logits = clusternet.forward(images)[0]
            neighbors_logits = clusternet.forward(neighbors)[0]

            # Calculating the SCAN-Loss
            total_loss, _, _ = scanloss.forward(anchors=anchors_logits, neighbors=neighbors_logits)

            '''anchors_id = clusternet.forward(images_u_id, forward_pass='return_all')
            anchors_clr = clusternet.forward(images_u_clr, forward_pass='return_all')
            neighbors_id = clusternet.forward(neighbor_images_id, forward_pass='default')
            scan_loss, _, _ = ScanLoss.forward(anchors=torch.cat([anchors_id['output'][0], anchors_id['output'][0]]), neighbors=torch.cat([neighbors_id, anchors_clr['output'][0]]))

            zs_indices = torch.where(masked_Ids == -1)[0]
            known_indices = torch.where(masked_Ids != -1)[0]

            F_zs = anchors_id['features'][zs_indices, :]
            P_zs = F.softmax(anchors_id['output'][0][zs_indices, :], dim=1)[:, zs_Ids]

            confident_indices = torch.topk(P_zs, k=K, dim=0, sorted=False)[1]

            Q = F_zs[confident_indices.T.flatten(), :].reshape(num_zs_Ids, K, 512)
            zs_means = Q.mean(dim=1)

            zs_similarities = torch.matmul(F_zs, zs_means.T)
            closest_examples = torch.topk(zs_similarities, k=K, dim=0, sorted=False)[1].transpose(1, 0)
            closest_examples = zs_indices[closest_examples.flatten()].reshape(num_zs_Ids, K)

            one_hot_matrix = torch.zeros((masked_Ids.shape[0], num_classes), dtype=torch.float, device=device)
            one_hot_matrix[known_indices, Ids[known_indices]] = 1.
            one_hot_matrix[closest_examples.flatten(), dummy.flatten()] = 1.

            mask = torch.where((one_hot_matrix != 0).any(dim=1))[0]
            class_weights = one_hot_matrix.sum(dim=0)

            logits = anchors_clr['output'][0]
            spice_loss = F.cross_entropy(logits[mask, :], one_hot_matrix[mask, :], weight=class_weights, reduction='mean')

            total_loss = scan_loss + 10**(-1) * spice_loss
            
            total_loss = total_loss/num_gradient_steps
            total_loss.backward()
            if ((i + 1) % num_gradient_steps == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += total_loss.detach().cpu().item()'''

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch%5) == 0:
            clusternet.eval()
            true_labels = []        # a torch tensor containing the true labels 
            predictions = []        # a torch tensor containing the predictions
            embeddings = []         # a torch tensor containing the ResNet embeddings, Used to find the correct nearest neighbors
            with torch.no_grad():
                for i, (images_batch, _, _, labels_batch, _) in enumerate(dataloader):
                    images_batch = images_batch.to(device)
                    images_batch = images_batch
                    x = clusternet.forward(images_batch, forward_pass='return_all')     # dictionary containing the embeddings and the logits for a batch
                    embeddings.append(x['features'].cpu())
                    batch_probs = F.softmax(x['output'][0], dim=1)      # each row corresponds the probabilites of a sample (the columns are as many as the classes)
                    
                    batch_predictions = torch.argmax(batch_probs, dim=1) # The prediction is the maximum of each row
                    predictions.append(batch_predictions.cpu())
                    true_labels.append(labels_batch)
                
            embeddings = torch.cat(embeddings, dim=0)
            indices = find_indices_of_closest_embeddings(embedings=F.normalize(embeddings, dim=1))
            true_labels = torch.cat(true_labels, dim=0)
            predictions = torch.cat(predictions, dim=0)
            correct_neighbors_mean(Ids=true_labels, indices=indices)
            nmi, ari, acc = cluster_metric(label=true_labels.numpy(), pred=predictions.numpy())
            print('------------------- Epoch: ', epoch,' ---------------------')
            # Print the evaluation metrics
            print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
            print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
            print(f"Accuracy (ACC): {acc:.2f}%")
            print('\n')

            earlystopping(val_accuracy=acc)
            
            torch.save(clusternet.backbone.state_dict(), 'NeuralNets/backbone_AwA2_224.pth')
            torch.save(clusternet.cluster_head.state_dict(), 'NeuralNets/cluster_head_AwA2_224.pth')
            if earlystopping.early_stop == True:
                break
            

def spice_training(num_epochs: int=200, num_classes: int=50):
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=2)
    clusternet = load_scan_trained_model(n_classes=50)
    clusternet = FreezeResnet(clusternet)

    littlenn = LittleNet()

    id_aug = Identity_Augmentation()
    weak_aug = Weak_Augmentation()
    strong_aug = RandAugment_Augmentation()

    known_Ids = (dataset.known_Ids).to(device)
    zs_Ids = (dataset.zs_Ids).to(device)
    num_zs_Ids = zs_Ids.numel()
    clusternet.to(device)
    littlenn.to(device)

    optimizer = optim.Adam(littlenn.parameters(), lr=10**(-4), weight_decay=10**(-4))
    earlystopping = EarlyStopping(patience=3, delta=0.01)

    K = 30
    dummy = zs_Ids.repeat(K, 1).T

    for epoch in range(0, num_epochs):
        for i, (images, _, Ids, masked_Ids) in enumerate(dataloader):
            images = images.to(device)
            Ids = Ids.to(device)
            masked_Ids = masked_Ids.to(device)

            zs_indices = torch.where(masked_Ids == -1)[0]
            known_indices = torch.where(masked_Ids != -1)[0]
            zs_images = images[zs_indices, :]

            F_zs = clusternet.forward(id_aug(zs_images), forward_pass='backbone')
            #P_zs = F.softmax(clusternet.forward(weak_aug(zs_images), forward_pass='default')[0], dim=1)[:, zs_Ids]
            P_zs = F.softmax(littlenn(clusternet.forward(weak_aug(zs_images), 'backbone')), dim=1)[:, zs_Ids]
            confident_indices = torch.topk(P_zs, k=K, dim=0, sorted=False)[1]

            Q = F_zs[confident_indices.T.flatten(), :].reshape(num_zs_Ids, K, 512)
            zs_means = Q.mean(dim=1)

            zs_similarities = torch.matmul(F_zs, zs_means.T)
            closest_examples = torch.topk(zs_similarities, k=K, dim=0, sorted=False)[1].transpose(1, 0)
            closest_examples = zs_indices[closest_examples.flatten()].reshape(num_zs_Ids, K)
            one_hot_matrix = torch.zeros((Ids.shape[0], num_classes), dtype=torch.float, device=device)
            one_hot_matrix[known_indices, Ids[known_indices]] = 1.
            one_hot_matrix[closest_examples.flatten(), dummy.flatten()] = 1.

            mask = torch.where((one_hot_matrix != 0).any(dim=1))[0]
            class_weights = one_hot_matrix.sum(dim=0)
            #logits = clusternet.forward(strong_aug(images), forward_pass='default')[0]
            logits = littlenn(clusternet.forward(strong_aug(images), 'backbone'))

            loss = F.cross_entropy(logits[mask,:], one_hot_matrix[mask,:], weight=class_weights, reduction='mean')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch%5) == 0:
            clusternet.eval()
            true_labels = []
            predictions = []
            with torch.no_grad():
                for i, (images_batch, _, labels_batch, _) in enumerate(dataloader):
                    images_batch = id_aug(images_batch.to(device))

                    resnet_embeddings = clusternet.forward(images_batch, 'backbone')
                    logits = littlenn(resnet_embeddings)
                    batch_probs = F.softmax(logits, dim=1)
                    
                    batch_predictions = torch.argmax(batch_probs, dim=1)
                    predictions.append(batch_predictions.cpu())
                    true_labels.append(labels_batch)
                
            true_labels = torch.cat(true_labels, dim=0)
            predictions = torch.cat(predictions, dim=0)
            nmi, ari, acc = cluster_metric(label=true_labels.numpy(), pred=predictions.numpy())
            print('------------------- Epoch: ', epoch,' ---------------------')
            # Print the evaluation metrics
            print(f"Normalized Mutual Information (NMI): {nmi:.2f}%")
            print(f"Adjusted Rand Index (ARI): {ari:.2f}%")
            print(f"Accuracy (ACC): {acc:.2f}%")
            print('\n')


def spice_training_features(num_epochs: int=200, num_classes: int=50):
    dataset_features = AwA2dataset_features()
    dataloader = DataLoader(dataset_features, batch_size=2048, shuffle=True, num_workers=2)
    littlenn = LittleNet(backbone_dim=2048)

    known_Ids = dataset_features.known_Ids.to(device)
    zs_Ids = dataset_features.zs_Ids.to(device)

    num_zs_Ids = zs_Ids.numel()
    littlenn.to(device)

    optimizer = optim.Adam(littlenn.parameters(), lr=10**(-4), weight_decay=10**(-4))
    earlystopping = EarlyStopping(patience=3, delta=0.01)
    K = 30
    dummy = zs_Ids.repeat(K, 1).T
    littlenn.double()
    for epoch in range(0, num_epochs):
        for i, (features, Ids, masked_Ids) in enumerate(dataloader):
            features = features.to(device)
            Ids = Ids.to(device)
            masked_Ids = masked_Ids.to(device)

            zs_indices = torch.where(masked_Ids == -1)[0]
            known_indices = torch.where(masked_Ids != -1)[0]

            zs_features = features[zs_indices,:]
            logits = littlenn.forward(features.double(), forward_pass='default')
            P_zs = F.softmax(logits[zs_indices, :], dim=1)[:, zs_Ids]
            confident_indices = torch.topk(P_zs, k=K, dim=0, sorted=False)[1]

            Q = zs_features[confident_indices.T.flatten(), :].reshape(num_zs_Ids, K, 2048)
            zs_means = Q.mean(dim=1)

            zs_similarities = torch.matmul(zs_features, zs_means.T)
            closest_examples = torch.topk(zs_similarities, k=K, dim=0, sorted=False)[1].transpose(1, 0)
            closest_examples = zs_indices[closest_examples.flatten()].reshape(num_zs_Ids, K)

            one_hot_matrix = torch.zeros((Ids.shape[0], num_classes), dtype=torch.float, device=device)
            one_hot_matrix[known_indices, Ids[known_indices]] = 1.
            one_hot_matrix[closest_examples.flatten(), dummy.flatten()] = 1.

            mask = torch.where((one_hot_matrix != 0).any(dim=1))[0]
            class_weights = one_hot_matrix.sum(dim=0)
            
            loss = F.cross_entropy(logits[mask, :], one_hot_matrix[mask, :], weight=class_weights, reduction='mean')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch%5) == 0:
            littlenn.eval()
            true_labels = []
            predictions = []
            with torch.no_grad():
                for i, (features, labels_batch, _) in enumerate(dataloader):
                    features = features.to(device)
                    logits = littlenn(features)
                    batch_probs = F.softmax(logits, dim=1)

                    batch_predictions = torch.argmax(batch_probs, dim=1)
                    predictions.append(batch_predictions.cpu())
                    true_labels.append(labels_batch)

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
                torch.save(littlenn.state_dict(), 'NeuralNets/littleNN.pth')
                break


#spice_training_features(num_epochs=200, num_classes=50)
#spice_training()


scan_training(num_epochs=200, num_classes=50)