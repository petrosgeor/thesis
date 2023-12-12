import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random
from utils import *
from models import *
from augmentations import *
import matplotlib.pyplot as plt
from scipy.special import binom

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
    def __init__(self, proportion = 1) -> None:
        self.proportion = proportion
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
        if self.proportion == None:
            return data, Ids, class2Id
        else:
            data, Ids = self.keep_part_of_dataset(data, Ids, self.proportion)
            return data, Ids, class2Id

    @staticmethod
    def keep_part_of_dataset(data: torch.Tensor, labels: torch.Tensor, proportion: int=1):
        assert (proportion <= 1) and (proportion > 0), 'proportion must be set to a number between 0 and 1'
        n_samples = labels.numel()
        n_samples_keep = int(n_samples * proportion)
        numbers = random.sample(range(n_samples), n_samples_keep)
        return data[numbers], labels[numbers]




class LinkedDataset(Dataset):
    '''
    contains the images that are Linked or can not link
    '''
    def __init__(self, cifardataset: CIFAR10, num_links: int = 1000):
        self.data, self.A_matrix, self.labels_subset = random_links2label(cifardataset.data, cifardataset.Ids, num_links=num_links)
        self.linked_indices = self.find_linked_indices()    # this list contains all the related images. e.g at [0] we have all the indices of A_matrix which are not 0
        self.related_images, self.relations, self.knowledge_list = self.organize_images()
        # find linked indices just as in SCAN

    def __getitem__(self, item):
        #return self.data[item], self.related_images[item], self.relations[item]
        image = self.data[item]
        image_related_indices = self.linked_indices[item]
        n_related_indices = image_related_indices.numel()
        assert n_related_indices >= 1, 'there is some image with no related indices, there must be a bug'
        random_index_index = torch.randint(0, n_related_indices, (1,)).item()
        random_column = image_related_indices[random_index_index]

        linked_image = self.data[random_column]
        relation = self.A_matrix[item, random_column]
        return image, linked_image, relation
    
    def __len__(self):
        return self.data.shape[0]


    def find_linked_indices(self):
        n_samples = self.A_matrix.shape[0]
        linked_indices = []         # linked_indices[0] contains all the columns of the A_matrix which are not 0 (related images to the first image)
        for i in range(0, n_samples):
            not_zero_columns = torch.where(self.A_matrix[i, :] != 0)[0]
            linked_indices.append(not_zero_columns)
        return linked_indices

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
        self.same_Ids_list = self.find_percentage_of_consistency()      # used to find the percentage of neighbors that share the same Id
        self.correct_links_list = self.find_correct_links_in_neighborhood()

    def __getitem__(self, item):
        '''
        returns an image, it's Id, and a random Neighbor
        '''
        related_indices = self.neighbor_indices[item, :]
        random_index_index = torch.randint(0, self.n_neighbors, (1,)).item()        # random index from the index tensor
        random_index = related_indices[random_index_index]
        return self.data[item,:], self.Ids[item], self.data[random_index, :]

    def __len__(self):
        return (self.Ids).numel()
    
    def find_percentage_of_consistency(self) -> list:
        same_Ids_list = []
        for i in range(self.Ids.numel()):
            Id = self.Ids[i]
            neighbor_indices = self.neighbor_indices[i, :]
            neighbor_Ids = self.Ids[neighbor_indices]
            same_Id = torch.where(neighbor_Ids == Id)[0].numel()
            same_Ids_list.append(same_Id)
        return same_Ids_list

    def find_correct_links_in_neighborhood(self) -> list:
        correct_links_list = []
        total_links = binom(self.n_neighbors, 2)
        for i in range(0, self.neighbor_indices.shape[0]):
            correct_links = 0
            neighbor_Ids = self.Ids[self.neighbor_indices[i, :]]
            Ids_count = torch.unique(neighbor_Ids, return_counts=True)[1]

            for j in range(0, Ids_count.shape[0]):
                if Ids_count[j].item() == 1:
                    continue
                
                correct_links += binom(Ids_count[j].item(), 2)
            correct_links_list.append(correct_links/total_links)
        return correct_links_list





#dataset = CIFAR10(proportion=1)
# n = dataset.__len__()
# numbers = random.sample(range(n), int(n/6))
# subset1 = Subset(dataset, indices=numbers)

#linked_dataset = LinkedDataset(dataset, num_links=1000)
# dataloader2 = DataLoader(linked_dataset, batch_size=100)

# for i, (images1, images2, relations) in enumerate(dataloader2):
#     print(images2.shape)

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
    






















