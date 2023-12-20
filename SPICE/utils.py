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



device = 'cuda'



class InfoNCELoss(nn.Module):
    def __init__(self, temperature: int)-> torch.Tensor:
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor)-> torch.Tensor:
        batch_size = z_i.shape[0]
        representations = torch.vstack((z_i, z_j))
        Sim_matrix = torch.matmul(representations, representations.T)/self.temperature
        Sim_matrix = torch.exp(Sim_matrix)

        n1 = torch.arange(0, batch_size)
        n2 = torch.arange(batch_size, 2*batch_size)

        mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        mask[n1, n2] = True
        mask[n2, n1] = True


        nominator_ij = Sim_matrix[n1, n2]
        nominator_ji = Sim_matrix[n2, n1]
        nominator = torch.cat((nominator_ij, nominator_ji), dim=0)
        denominator = (~mask * Sim_matrix).sum(dim=1)
        return torch.mean(-(nominator/denominator).log())


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(CustomCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted_probs:torch.Tensor, labels:torch.Tensor)-> torch.Tensor:
        '''
        predicted_probs: A torch 2d tensor. Each row represents probabilities for one cluster center
        labels: A 1d torch tensor
        '''
        one_hot_labels = self.CreateOneHotVectors(labels, num_classes=self.num_classes)
        Matrix = (-(one_hot_labels * predicted_probs.log())).sum(dim=1)
        return torch.mean(Matrix)

    def forwardOneHot(self, predicted_probs: torch.Tensor, one_hot_labels: torch.Tensor)-> torch.Tensor:
        Matrix = (-(one_hot_labels * predicted_probs.log())).sum(dim=1)
        return torch.mean(Matrix)

    def CreateOneHotVectors(self, labels: torch.Tensor, num_classes: int = 100)-> torch.Tensor:
        num_examples = labels.shape[0]
        one_hot = torch.zeros(num_examples, num_classes, device=str(labels.device))
        one_hot[torch.arange(0, num_examples), labels] = 1
        return one_hot



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
    def __call__(self, val_accuracy):


        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0




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

        


