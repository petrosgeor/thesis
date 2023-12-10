import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from utils import *
from models import *
from augmentations import *
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def plot_image_from_tensor(tensor):
    numpy_image = (tensor.permute(1,2,0).numpy()).astype(np.int32)
    #print(numpy_image)
    plt.figure()
    plt.imshow(numpy_image)
    plt.axis('off')
    plt.show()

def create_A_matrix(labels):
    size = labels.size(0)
    A_matrix = torch.zeros(size, size)

    for n in range(size):
        # Get a random index k that is different from n
        k = random.choice(range(len(labels)))
        while(k == n):
            k = random.choice(range(len(labels)))        
        if labels[n] == labels[k]:
            A_matrix[n, k] = 1
            A_matrix[k, n] = 1
        else:
            A_matrix[n, k] = -1
            A_matrix[k, n] = -1

    assert (torch.any(A_matrix != 0, dim=1)).all().item(), 'not all rows contain atleast one value which is not 0'
    assert (torch.any(A_matrix != 0, dim=0)).all().item(), 'not all columns contain atleast one value which is not 0'
    return A_matrix


# labels = torch.tensor([1,2,3,4, 1, 1])
# print(create_A_matrix(labels))

def random_links2label(data:torch.Tensor, labels: torch.Tensor, num_links: int):
    indices = torch.randint(low=0, high=len(labels), size=(num_links,))
    labels_subset = labels[indices]
    X_subset = data[indices]
    A_matrix = create_A_matrix(labels=labels_subset)
    return X_subset, A_matrix, labels_subset

'''labels = torch.tensor([1,1,2,3,1])
data = torch.randn((len(labels), 100))
print(random_links2label(data, labels, num_links=len(labels)))
'''

class CIFAR10(Dataset):
    def __init__(self) -> None:
        self.data, self.Ids, self.class2Id = self.load_data()
        
    def __getitem__(self, item):
        return self.data[item,:], self.Ids[item]

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




class LinkedDataset(Dataset):
    '''
    contains the images that are Linked or can not link
    '''
    def __init__(self, cifardataset: CIFAR10, num_links: int = 1000):
        self.data, self.A_matrix, self.labels_subset = random_links2label(cifardataset.data, cifardataset.Ids, num_links=num_links)
        self.related_images, self.relations, self.knowledge_list = self.organize_images()

    def __getitem__(self, item):
        return self.data[item], self.related_images[item], self.relations[item]
    
    def __len__(self):
        return self.data.shape[0]

    def organize_images(self):
        n_images = self.data.shape[0]
        knowledge_list = []             # list of the same length as A_matrix.shape[0]. knowledge_list[n] contains the indices of A_matrix[n,:] where
        relations = []                  # it's not 0
        related_images = []
        for i in range(0, n_images):
            indices = torch.where(self.A_matrix[i, :] != 0)[0]
            knowledge_list.append(indices)
            relations.append(self.A_matrix[i, indices])
            related_images.append(self.data[indices])
        assert self.check_values(relations), 'not all tensors in the relations list contain only -1 and 1. there are other values'
        related_images, relations = self.pad_tensors(relations, related_images)
        return related_images, relations, knowledge_list

    @staticmethod
    def check_values(tensor_list: list):
        for tensor in tensor_list:
            unique_values = torch.unique(tensor)
            if len(unique_values) > 2 or not all(value.item() in [-1, 1] for value in unique_values):
                return False
        return True

    @staticmethod
    def pad_tensors(relations: list, related_images: list) -> tuple:
        max_length = max(len(tensor) for tensor in relations)
        padded_relations = []
        for tensor in relations:
            padding = max_length - len(tensor)
            padded_tensor = torch.cat((tensor, torch.full((padding,), float('nan'))))
            padded_relations.append(padded_tensor)

        max_first_dim = max(tensor.shape[0] for tensor in related_images)
        related_images_padded = []
        for tensor in related_images:
            padding = max_first_dim - tensor.shape[0]
            nan_padding = torch.empty((padding,) + tensor.shape[1:]).fill_(float('nan'))
            padded_tensor = torch.cat((tensor, nan_padding))
            related_images_padded.append(padded_tensor)
        return related_images_padded, padded_relations



class SCANdatasetWithNeighbors(Dataset):
    def __init__(self, data: torch.Tensor, Ids: torch.Tensor, neighbor_indices: torch.Tensor):
        self.data = data
        self.Ids = Ids
        self.neighbor_indices = neighbor_indices
        self.n_neighbors = neighbor_indices.shape[1]

    def __getitem__(self, item):
        return self.data[item,:], self.Ids[item], self.data[self.neighbor_indices[item, :], :]

    def __len__(self):
        return (self.Ids).numel()





# dataset = CIFAR10()
# linked_dataset = LinkedDataset(dataset, num_links=1000)



# rows_with_1 = torch.where(torch.vstack(linked_dataset.relations) == 1)



# plot_image_from_tensor(linked_dataset.data[15])
# plot_image_from_tensor(linked_dataset.related_images[15][1])
# print(linked_dataset.relations[15])




# dataloader1 = DataLoader(dataset, batch_size=1000)
# dataloader2 = DataLoader(linked_dataset, batch_size=100)
# aug = SimCLRaugment()

# num_epochs = 3
# for epoch in range(num_epochs):
#     dataloader_iterator = iter(dataloader2)
#     for i, X in enumerate(dataloader1):
#         try:
#             image, related_images, relations = next(dataloader_iterator)
#         except StopIteration:
#             dataloader_iterator = iter(dataloader2)
#             image, related_images, relations = next(dataloader_iterator)
#             print(relations.shape)
    






















