import torch
import numpy as np
from dataset import *
from utils import *
from augmentations import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from evaluation import *

def plot_image_from_tensor(tensor):
    numpy_image = tensor.permute(1,2,0).numpy()
    plt.imshow(numpy_image)
    plt.axis('off')
    #plt.show()
    plt.savefig('plots/aaaaa')


device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

x = resnet18()
resnet = x['backbone']
resnet.load_state_dict(torch.load('NeuralNets/backbone_AwA2.pth'))

clusternet = ClusteringModel(backbone={'backbone':resnet, 'dim': 512}, nclusters=50)
clusternet.cluster_head.load_state_dict(torch.load('NeuralNets/cluster_head_AwA2.pth'))
dataset = AwA2dataset()
id_aug = Identity_Augmentation()

dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
clusternet.to(device)
predictions = []
embeddings = []
labels = []
clusternet.eval()
with torch.no_grad():
    for i, (X_batch, _, labels_batch, _) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        X_batch = id_aug(X_batch)
        

        x = clusternet.forward(X_batch, forward_pass='return_all')
        embeddings.append(x['features'].cpu())
        batch_probs = F.softmax(x['output'][0], dim=1)

        batch_predictions = torch.argmax(batch_probs, dim=1)

        predictions.append(batch_predictions.cpu())
        labels.append(labels_batch)


predictions = torch.cat(predictions, dim=0)
labels = torch.cat(labels, dim=0)
predictions2labels = torch.from_numpy(get_y_preds(y_true=labels.numpy(), cluster_assignments=predictions.numpy(), n_clusters=50))
embeddings = torch.cat(embeddings, dim=0)
embeddings = F.normalize(embeddings, dim=1)

Ids = torch.unique(predictions, sorted=True)
means = []
for i, id in enumerate(Ids):
    indices = torch.where(predictions == id)[0]
    m = embeddings[indices, :].mean(dim=0).unsqueeze(0)
    means.append(m)

means = torch.cat(means, dim=0)
similarity_matrix = torch.matmul(means, embeddings.T)
best_examples = torch.topk(similarity_matrix, k=10, dim=1)[1]

known_Ids = dataset.known_Ids
zs_Ids = dataset.zs_Ids

known_Ids2clusterassignments = {}
for i in known_Ids:
    indices = torch.where(labels == i)[0]
    x = torch.unique(predictions[indices], return_counts=True)
    print(i)
    print(x)
    print('------------------------')
    clusters = x[0]
    counts = x[1]
    c = torch.argmax(counts)
    known_Ids2clusterassignments[i.item()] = clusters[c].item()

known_clusters = torch.tensor(list(known_Ids2clusterassignments.values()))
#zs_clusters = torch.tensor(list(set(list(range(50))) - set(known_clusters.tolist())))

zs_clusters = torch.arange(0, known_Ids.numel() + zs_Ids.numel())
zs_clusters = torch.masked_select(zs_clusters, ~torch.isin(zs_clusters, known_clusters))

def plot_histogram_backbone_NN():
    #clusternet = initialize_clustering_net(n_classes=50, nheads=1)
    id_aug = Identity_Augmentation()
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=60, shuffle=False)
    embeddings = []
    Ids = []
    clusternet.to(device)
    clusternet.eval()
    with torch.no_grad():
        for i, (X_batch, _, labels_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            X_batch = id_aug(X_batch)
            embeddings_batch = clusternet.forward(X_batch, forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())
            Ids.append(labels_batch)

        embeddings = torch.cat(embeddings, dim=0)
        Ids = torch.cat(Ids, dim=0)
        indices = find_indices_of_closest_embeddings(F.normalize(embeddings, dim=1), n_neighbors=20)
        #Ids = dataset.Ids
        correct = []
        for i, id in enumerate(Ids):
            neighbor_indices = indices[i, :]    
            n_correct = torch.where(Ids[neighbor_indices] == id)[0].numel()
            correct.append(n_correct)
        
        plt.figure(figsize=(8, 6))  # Set the figure size (optional)
        plt.hist(correct, bins=20, color='skyblue', edgecolor='black')  
        plt.title('Histogram of 20 Distinct Values') 
        plt.xlabel('Values') 
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.5)
        plt.show()


#plot_histogram_backbone_NN()










