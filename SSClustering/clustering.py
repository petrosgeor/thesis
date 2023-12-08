import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.optim as optim
from sklearn import metrics

device = 'cuda'

def calculate_NMI(predictions: np.ndarray, true_labels: np.ndarray, n_clusters: int = 10) -> int:
    return metrics.normalized_mutual_info_score(labels_true=true_labels, labels_pred=predictions)


class ClusterHead(nn.Module):
    def __init__(self, n_features_in, num_classes = 10):
        super(ClusterHead, self).__init__()
        self.n_features_in = n_features_in
        self.num_classes = num_classes
        self.cluhead = nn.Sequential(
            nn.BatchNorm1d(n_features_in),  # a particular cluster
            nn.ReLU(),
            nn.Linear(n_features_in, n_features_in),
            nn.BatchNorm1d(n_features_in),
            nn.ReLU(),
            nn.Linear(n_features_in, self.num_classes)
        )

    def forward(self, x):
        probs = self.cluhead(x)
        probs = F.softmax(probs, dim=1)
        return probs


class ClusterDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, n_neighbors: int=20):
        self.embeddings = embeddings
        self.labels = labels
        self.n_features = embeddings.shape[1]
        self.n_neighbors = n_neighbors
        self.indices = self.find_nearest_embeddings()

    def find_nearest_embeddings(self):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.embeddings)
        _, indices = nbrs.kneighbors(self.embeddings)
        return indices

    def __getitem__(self,item):
        return self.embeddings[item], self.embeddings[self.indices[item,:], :], self.labels[item]
    
    def __len__(self):
        return len(self.labels)


class ClusterConsistencyLoss(nn.Module):
    def __init__(self):
        super(ClusterConsistencyLoss, self).__init__()
    def forward(self, probs, probs_neighbors):
        inner_products = torch.bmm(probs_neighbors, probs.unsqueeze(1).transpose(1,2))
        inner_products = inner_products.squeeze()
        l = torch.sum(inner_products.log(), dim=1)
        return -torch.mean(l)

class ClusterEntropyLoss(nn.Module):
    def __init__(self):
        super(ClusterEntropyLoss, self).__init__()
    def forward(self, probs: torch.Tensor):
        priors = torch.mean(probs, dim=0)
        return torch.sum(priors * priors.log())

def train_cluster_head(embeddings: np.ndarray, labels: np.ndarray, n_neighbors: int=20):
    dataset = ClusterDataset(embeddings=embeddings, labels=labels, n_neighbors=n_neighbors) 
    dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
    num_classes = len(np.unique(labels))
    #print('1')
    ClusterNet = ClusterHead(n_features_in=dataset.n_features).to(device)
    ClusterNet.float()
    constistency_criterion = ClusterConsistencyLoss()
    optimizer = optim.Adam(ClusterNet.parameters(), lr=10**(-3))
    entropy_criterion = ClusterEntropyLoss()
    for epoch in range(0, 50):
        for i, (embeddings, neighbor_embeddings, _) in enumerate(dataloader):
            embeddings = embeddings.to(device)
            n_samples = embeddings.shape[0]
            n_features = embeddings.shape[1]
            neighbor_embeddings = neighbor_embeddings.to(device)
            #print(neighbor_embeddings.shape)

            probs = ClusterNet(embeddings.float())
            probs_neighbors = ClusterNet(((neighbor_embeddings.reshape(n_samples*n_neighbors, n_features))).float())
            #print(probs_neighbors.shape)
            probs_neighbors = probs_neighbors.reshape(n_samples, n_neighbors, num_classes)
            #print((loss(probs, probs_neighbors)).shape)
            loss = constistency_criterion(probs, probs_neighbors) + entropy_criterion(probs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    with torch.no_grad():
        predictions = []
        true_labels = []
        for i, (batch_embeddings, _, labels) in enumerate(dataloader):
            batch_embeddings = batch_embeddings.to(device)
            probs = ClusterNet(batch_embeddings.float())
            predictions.append(torch.argmax(probs, dim=1))
            true_labels.append(labels)

        predictions = torch.cat(predictions, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        print('the NMI is: ', calculate_NMI(predictions.cpu().numpy(), true_labels.numpy()))




'''x = np.random.randn(1000, 128)
labels = np.random.randint(0, 10, size=(1000, ))
train_cluster_head(x, labels)

x = torch.tensor([[[1,2,3],[10,20,30]], [[-1,-2,-3],[-10,-20,-30]]])


'''


