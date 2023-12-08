import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from utils import *
from augmentations import *



class CIFAR10(Dataset):
    def __init__(self) -> None:
        self.data, self.Ids, self.class2Id = self.load_data()
        
    def __getitem__(self, item):
        return self.data[item,:]#, self.Ids[item]

    def __len__(self):
        return len(self.Ids)

    def load_data(self):
        classes_list = []
        path = find_environment()
        data = torch.from_numpy(np.load(path + 'data.npy'))
        data = torch.permute(data, (0, 3,1,2))
        Ids = torch.from_numpy(np.load(path + 'Ids.npy'))

        with open(path + 'class2Id.pkl', 'rb') as file:
            class2Id = pickle.load(file)
        
        return data, Ids, class2Id


def create_labeled_dataset(dataset: CIFAR10, num_labeled: int = 200):
    random_indices = np.random.choice(a = len(dataset.Ids), size=num_labeled, replace=False)
    random_indices = torch.from_numpy(random_indices)
    X_subset = dataset.data[random_indices]
    labels = dataset.Ids[random_indices]
    return X_subset, labels



class LabeledSubset(Dataset):
    def __init__(self, dataset:CIFAR10,num_labeled: int):
        self.data, self.labels = create_labeled_dataset(dataset, num_labeled=num_labeled)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

'''dataset = CIFAR10()
subset = LabeledSubset(dataset, num_labeled=200)

'''















