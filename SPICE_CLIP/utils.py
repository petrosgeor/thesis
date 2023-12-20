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

def running_on_colab():
    import os
    return 'COLAB_GPU' in os.environ


def find_environment():

    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/cifar-100/'
    else:
        path = 'cifar-100/'
    return path


def SavePretrainedNeuralNet(neural_net, epoch, current_lr):
    feature_dim = neural_net.feature_dim
    num_classes = neural_net.cluster_num
    if running_on_colab() == False:
        path = 'Neural_Networks/PretrainingResults/'
    else:
        path = '/content/drive/MyDrive/data/Neural_Networks/PretrainingResults/'

    state = {
        'epoch': epoch,
        'network_state': neural_net.state_dict(),
        'feature_dim': feature_dim,
        'num_classes': num_classes,
        'current_lr': current_lr
    }
    torch.save(state, path + 'PretrainedModelStateDict.pth')


def LoadPretrainedNeuralNet(resnet='resnet18'):
    if running_on_colab() == False:
        path = 'Neural_Networks/PretrainingResults/'
    else:
        path = '/content/drive/MyDrive/data/Neural_Networks/PretrainingResults/'
    state = torch.load(path + 'PretrainedModelStateDict.pth')

    num_classes = state['num_classes']
    resnet, hidden_dim = get_resnet(model=resnet)
    network = Network(resnet, hidden_dim, feature_dim=128, class_num=num_classes)
    current_lr = state['current_lr']

    epoch = state['epoch']
    network.load_state_dict(state['network_state'])

    return network, epoch, current_lr


def LoadEMtrainedNeuralNet(resnet='resnet18', num_classes: int=100)-> Network:
    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/Neural_Networks/'
    elif running_on_colab() == False:
        path = 'Neural_Networks/'
    
    resnet, hidden_dim = get_resnet(model=resnet)
    network = Network(resnet, hidden_dim, feature_dim=128, class_num=num_classes)
    network.load_state_dict(torch.load(path + 'EMresults/EMtrainedNetStateDict.pth'))
    return network

def SaveEMtrainedNeuralNet(model: Network)-> None:
    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/Neural_Networks/'
    elif running_on_colab() == False:
        path = 'Neural_Networks/'
    torch.save(model.state_dict(), path + 'EMresults/EMtrainedNetStateDict.pth')


    

def ChangeFinalLayerDim(network: Network, new_output_dim: int)-> Network:
    '''Changes the dimension of the final layer of the neural network (this is used for testing)'''
    in_features = network.cluster_projector[5].in_features
    network.cluster_projector[5] = nn.Linear(in_features, new_output_dim)
    network.cluster_num = new_output_dim 
    return network




def SaveLogRegrAccuracy(accuracy):

    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/Neural_Networks/'
    else:
        path = 'Neural_Networks/'

    try:
        with open(path + 'LogRegrAccuracy.pkl', 'rb') as file:
            loaded_list = pickle.load(file)
            loaded_list.append(accuracy)
        with open(path + 'LogRegrAccuracy.pkl', 'w') as file:
            pickle.dump(loaded_list, file)
    except:
        loaded_list = [accuracy]
        with open(path + 'LogRegrAccuracy.pkl', 'w') as file:
            pickle.dump(loaded_list, file)




def set_device():
    if running_on_colab() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    return device

device = set_device()
device = 'cuda'

class CustomDataloader(DataLoader):
    def __init__(self, known_Ids: np.ndarray, zs_Ids: np.ndarray, dataset: Any, batch_size=128, shuffle = True, **kwargs):
        super(CustomDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.known_Ids = (torch.tensor(known_Ids, dtype=torch.uint8)).type(torch.int)
        self.zs_Ids = (torch.tensor(zs_Ids, dtype=torch.uint8)).type(torch.int)




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


def FreezeResnet(model: Network)-> Network:
    for name, param in model.resnet.named_parameters():
        param.requires_grad = False
    return model


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
        if running_on_colab() == True:
            path = '/content/drive/MyDrive/data/Neural_Networks/EMresults'   
        elif running_on_colab() == False:
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
        if running_on_colab() == True:
            path = '/content/drive/MyDrive/data/Neural_Networks/EMresults/'   
        elif running_on_colab() == False:
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
        num_classes = np.unique(labels_pred)

        classes = [str(i) for i in range(len(num_classes))]  # Replace with your class names
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

        

def ReliablePseudoLabeling(net: Network, dataloader: CustomDataloader, Ns: int, lamda: int) -> torch.Tensor:
    '''
    Returns the generated pseudo labels
    Args: 
        net: a neural network
        dataloader: my implementation of a custom dataloader
        Ns: Number of neighbors to consider
        lamda: threshold value
    '''
    assert net.cluster_projector[-1].out_features == dataloader.dataset.zs_Ids.shape[0] + dataloader.dataset.known_Ids.shape[0], 'the \
        output dimension of the neural network does not match the number of training classes'

    assert dataloader.dataset.pretrain == False, 'the defined dataset should have the pretrain parameter switched to False'

    predictions = []
    embeddings = []
    true_labels = []
    net.to(device)
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(dataloader):
            X, _, _, ids, _ = data
            X = X.to(device)

            batch_embeddings = net.forward_r(X)
            Probs = net.forward_c(X)
            batch_preds = torch.argmax(Probs, dim=1)

            true_labels.append(ids)
            predictions.append(batch_preds.cpu())
            embeddings.append(batch_embeddings.cpu())
        
        true_labels = torch.cat(true_labels, dim=0)     # the true labels of the dataset
        predictions = torch.cat(predictions, dim=0)
        embeddings = torch.cat(embeddings, dim=0)

        assert torch.all(true_labels != -1), 'there are labels that equal -1. Possibly the dataset.pretrain parameter should be swithed to False'

        pseudo_labels = torch.full_like(predictions, fill_value = -1)
        Dist_matrix = torch.matmul(embeddings, embeddings.T)    # Matrix of shape N x N where N is the total number of samples in the dataset
        closest_indices = torch.topk(Dist_matrix, k=Ns, dim=1, largest=True, sorted=False)[1]   # Tensor of shape (n_samples, Ns), each row contains indices

        for i in range(pseudo_labels.numel()):
            instance_inces = closest_indices[i, :]
            purity = (predictions[i] == predictions[instance_inces]).numel()/Ns
            if purity >= lamda:
                pseudo_labels[i] = predictions[i]

        fm_indices = torch.where(pseudo_labels != -1)[0]
        ClusteringMetrics.plot_confusion_matrix(true_labels[fm_indices].numpy(), pseudo_labels[fm_indices].numpy())
        return pseudo_labels
    





