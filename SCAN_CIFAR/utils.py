from typing import Any
import torch
from models import *
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin
import copy


device = 'cuda'




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_accuracy, model):
        if self.best_score is None:
            self.best_score = val_accuracy
            self.best_model = copy.deepcopy(model)
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_best_model()
        else:
            self.best_score = val_accuracy
            self.counter = 0
            self.best_model = copy.deepcopy(model)
            self.save_best_model()

    def save_best_model(self):
        torch.save(self.best_model.state_dict(), self.path)

class Cluster_KL(nn.Module):
    def __init__(self, num_classes: int, target_distribution: torch.Tensor) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.target_distribution = target_distribution

    def forward(self, P: torch.Tensor) -> torch.Tensor: 
        batch_samples = P.shape[0]      # number of samples in the batch
        emperical_distribution = torch.sum(P, dim=0)/batch_samples
        return (emperical_distribution * (emperical_distribution/self.target_distribution).log()).sum()


class ClusteringMetrics(object):
    def __init__(self, num_classes:int, seen_classes: torch.Tensor, zs_classes: torch.Tensor, saving_file='cluter_results/'):
        self.num_classes = num_classes
        self.seen_classes = seen_classes.cpu().numpy()
        self.zs_classes = zs_classes.cpu().numpy()
        self.saving_file = saving_file

    def __call__(self, labels_true: torch.Tensor, labels_pred: torch.Tensor) -> None:
        #self.conf_matrix(labels_pred, labels_true)
        np_labels_true = labels_true.numpy()
        np_labels_pred = labels_pred.numpy()

        NMI_score = ClusteringMetrics.Normalized_Mutual_Info_score(np_labels_true, np_labels_pred)
        accuracy_score = ClusteringMetrics.Accuracy_score(np_labels_true, np_labels_pred)
        zs_accuracy_score = ClusteringMetrics.ZSAccuracyScore(np_labels_true, np_labels_pred, zs_classes=self.zs_classes)
        ClusteringMetrics.save_results(NMI_score, accuracy_score, zs_accuracy_score)
        
    @staticmethod
    def Normalized_Mutual_Info_score(labels_true: np.ndarray, labels_pred: np.ndarray)-> int:
        return metrics.normalized_mutual_info_score(labels_true, labels_pred)
    
    @staticmethod
    def Accuracy_score(labels_true: np.ndarray, labels_pred: np.ndarray)-> int:
        return metrics.accuracy_score(labels_true, labels_pred)

    @staticmethod
    def ZSAccuracyScore(labels_true:np.ndarray, labels_pred: np.ndarray, zs_classes: np.ndarray)-> int:
        unknown_indices_true = np.where(np.isin(labels_true, zs_classes))[0]
        preds = labels_pred[unknown_indices_true]
        true_labels = labels_true[unknown_indices_true]
        return metrics.accuracy_score(true_labels, preds)
    
    
    @staticmethod
    def save_results(nmi: int, accuracy: int, zs_accuracy: int) -> None:
        path = 'Neural_Networks/EMresults'     

        if len(os.listdir(path)) != 0:
            df = pd.read_csv(path + '/clustering_results.csv')
            new_row = {'NMI': nmi, 'Accuracy': accuracy, 'ZS_accuracy': zs_accuracy}
            df.loc[len(df)] = new_row
            df.to_csv(path + '/clustering_results.csv', index=False)

        elif len(os.listdir(path)) == 0:
            new_row = [{'NMI': nmi, 'Accuracy': accuracy, 'ZS_accuracy': zs_accuracy}]
            df = pd.DataFrame(new_row)
            df.to_csv(path + '/clustering_results.csv', index=False)
            
    
    @staticmethod
    def plot_clustering_results()-> None:
        path = 'Neural_Networks/EMresults/'
        df = pd.read_csv(path + 'clustering_results.csv')

        data = df.to_numpy()
        fig, ax = plt.subplots(3, 1, figsize=(8,6))

        ax[0].plot(data[:, 0])
        ax[0].set_ylabel('NMI')

        ax[1].plot(data[:, 1])
        ax[1].set_ylabel('Accuracy')

        ax[2].plot(data[:, 2])
        ax[2].set_ylabel('Accuracy on zs classes')

        ax[2].set_xlabel('Epoch')

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> None:
        conf_matrix = metrics.confusion_matrix(labels_true, labels_pred)
        plt.figure(figsize=(8, 6))

        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = [str(i) for i in range(20)]  # Replace with your class names
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()


def map_back_indices(selected_indices: torch.Tensor, selected_indices2) -> torch.Tensor:
    return selected_indices[selected_indices2]

# x = torch.tensor([1,10,20,-1,5])
# y = torch.tensor([3,1,0])
# print(x[y])



def find_indices_of_closest_embeddings(embedings: torch.Tensor, masked_Ids: torch.Tensor,
                                        n_neighbors: int = 20, return_distances=False) -> torch.Tensor:
    D = torch.matmul(embedings, embedings.T)
    n_samples = masked_Ids.numel()
    #_, all_indices = torch.topk(D, k=n_neighbors, dim=1)
    known_dictionary = {}
    known_Ids = torch.unique(torch.masked_select(masked_Ids, masked_Ids != -1))
    for i in known_Ids:
        known_dictionary[i.item()] = torch.where(masked_Ids == i)[0]
    
    neighbor_indices = []
    for i in range(0, n_samples):
        if torch.isin(masked_Ids[i], known_Ids).item():
            neighbors = known_dictionary[masked_Ids[i].item()]
            neighbor_indices.append(neighbors)
        else:
            e = D[i, :]
            indices = torch.topk(e, k=n_neighbors)[1]
            neighbor_indices.append(indices)
    return neighbor_indices


def initializeClusterModel(n_heads: int = 1, dataset_name: str = 'cifar10', freeze_backbone=False):     # TRAINED USING THE SIMCLR ALGORITHM
    assert (dataset_name == 'cifar10') | (dataset_name == 'cifar100'), 'no implementation yet for the other datasets'
    backbone = resnet18()
    con_model = ContrastiveModel(backbone=backbone)
    file_path_10 = 'NeuralNets/simclr_cifar10.pth'
    file_path_100 = 'NeuralNets/simclr_cifar20.pth'
    if dataset_name == 'cifar10':
        n_clusters = 10
        checkpoint = torch.load(file_path_10)

        con_model.load_state_dict(checkpoint)

        clustermodel = ClusteringModel(backbone={'backbone': con_model.backbone, 'dim': con_model.backbone_dim}, nclusters=n_clusters)
        if freeze_backbone:
            for param in clustermodel.backbone.parameters():
                param.requires_grad = False
        return clustermodel
    elif dataset_name == 'cifar100':
        n_clusters = 20
        checkpoint = torch.load(file_path_100)
        con_model.load_state_dict(checkpoint)
        clustermodel = ClusteringModel(backbone={'backbone': con_model.backbone, 'dim': con_model.backbone_dim}, nclusters=n_clusters)
        if freeze_backbone:
            for param in clustermodel.backbone.parameters():
                param.requires_grad = False
        return clustermodel


def initializeSCANmodel(n_heads: int = 1, dataset_name: str = 'cifar100', freeze_backbone=False):   # TRAINED USING THE SCAN LOSS
    assert (dataset_name == 'cifar10') | (dataset_name == 'cifar100'), 'no implementation yet for the other datasets'
    backbone = resnet18()
    contrastive_model = ContrastiveModel(backbone=backbone)
    file_path_10 = 'NeuralNets/scan_cifar10.pth'
    file_path_100 = 'NeuralNets/scan_cifar20.pth'

    if dataset_name == 'cifar10':
        n_clusters = 10
        cluster_model = ClusteringModel(backbone= {'backbone': contrastive_model.backbone, 'dim': contrastive_model.backbone_dim}, nclusters=n_clusters)
        cluster_model.load_state_dict(torch.load(file_path_10))
        if freeze_backbone:
            for param in cluster_model.backbone.parameters():
                param.requires_grad = False
        return cluster_model
    elif dataset_name == 'cifar100':
        cluster_model = ClusteringModel(backbone= {'backbone': contrastive_model.backbone, 'dim': contrastive_model.backbone_dim}, nclusters=n_clusters)
        cluster_model.load_state_dict(torch.load(file_path_100))
        if freeze_backbone:
            for param in cluster_model.backbone.parameters():
                param.requires_grad = False
        return cluster_model






class GaussianMixture(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.means = []
        self.cov_matrices = []
        self.known_classes_num = None
    
    def fit(self, X: torch.Tensor, masked_Ids: torch.Tensor):
        known_Ids = torch.unique(torch.masked_select(masked_Ids, masked_Ids != -1))
        for i in known_Ids:
            indices = torch.where(masked_Ids == i)[0]
            mean = torch.mean(X[indices, :], dim=0)
            cov_matrix = torch.cov(X[indices, :].T)
            self.means.append(mean)
            self.cov_matrices.append(cov_matrix)
        self.known_classes_num = len(self.means)
        
    def predict(self, X: torch.Tensor):
        probs = []
        for i in range(0, self.known_classes_num):
            class_probs = self.GaussianPDFprob(X, mean=self.means[i], cov_matrix=self.cov_matrices[i])
            probs.append(class_probs)
        probs = torch.cat(probs, dim=0)
        total_probs = torch.mean(probs, dim=0)
        return total_probs

    @staticmethod
    def GaussianPDFprob(x: torch.Tensor, mean: torch.Tensor, cov_matrix: torch.Tensor):
        num_features = x.shape[1]
        A = x - mean
        C = torch.linalg.inv(cov_matrix)
        B = torch.matmul(C, A.T)

        R = torch.bmm(A.unsqueeze(1), B.T.unsqueeze(1).transpose(2,1)).squeeze()
        R = torch.exp(-1/2 * R) * (1/(2*torch.pi)**(num_features/2)) * (1/(torch.linalg.det(cov_matrix))**(1/2))
        return R
    

# labels = torch.randint(-1, 10, (10000,))
# X= torch.randn(10000, 100)
# model = GaussianMixture()
# model.fit(X, masked_Ids=labels)


