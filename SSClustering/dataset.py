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
import torch.nn.functional as F

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

def fill_A_matrix(A_matrix: torch.Tensor):
    '''
    if A[i,j] = 1 and A[j,k] = -1 then A[i, k] = -1 also
    '''
    n_samples = A_matrix.shape[0]
    for i in range(0, n_samples):
        y = torch.where(A_matrix[i, :] == 1)[0]
        if y.numel() == 0:
            continue

        for j in range(0, y.shape[0]):
            ii = y[j].item()
            idx_minus = torch.where(A_matrix[ii,:] == -1)[0]
            if idx_minus.numel() == 0:
                continue
            x = torch.full_like(idx_minus, fill_value=i)
            A_matrix[x, idx_minus] = -1
    return A_matrix



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
    #A_matrix = fill_A_matrix(A_matrix)
    return A_matrix


# labels = torch.tensor([1,2,3,1, 1, 1, 1, 1, 3])
# print(create_A_matrix(labels))

def random_links2label(data:torch.Tensor, labels: torch.Tensor, num_links: int):
    indices = torch.randint(low=0, high=len(labels), size=(num_links,))
    labels_subset = labels[indices]
    X_subset = data[indices]
    A_matrix = create_A_matrix(labels=labels_subset)
    return X_subset, A_matrix, labels_subset, indices


def create_big_A_matrix(labels: torch.Tensor, proportion_links: int, only_positives: bool):
    assert (proportion_links <= 2) and (proportion_links >= 0), 'the number of links is basically a proportion of the dataset'
    n_samples = labels.numel()
    n_links = int(proportion_links * n_samples)
    indices = torch.arange(0, n_samples, step=1).tolist()
    if only_positives == False:
        random_pairs = [random.choice(indices) for _ in range(2 * n_links)]
        #random_pairs = random.sample(indices, k=2*num_links)
        indices_pairs = [torch.tensor([random_pairs[i], random_pairs[i+1]]) for i in range(0, len(random_pairs), 2)]
        indices_pairs = torch.stack(indices_pairs, dim=0)

        #indices_pairs = torch.cat((indices_pairs, indices_pairs[:, [1,0]]), dim=0)
        relations = torch.eq(labels[indices_pairs[:, 0]], labels[indices_pairs[:, 1]])
        relations = relations.type(torch.float) * 2 - 1
        A_matrix = torch.sparse.FloatTensor(indices_pairs.T, relations, torch.Size([n_samples, n_samples]))
        return A_matrix
    else:
        random_pairs = [random.choice(indices) for _ in range(2 * n_links * 20)]
        #random_pairs = random.sample(indices, k=2*num_links)
        indices_pairs = [torch.tensor([random_pairs[i], random_pairs[i+1]]) for i in range(0, len(random_pairs), 2)]
        indices_pairs = torch.stack(indices_pairs, dim=0)

        #indices_pairs = torch.cat((indices_pairs, indices_pairs[:, [1,0]]), dim=0)
        relations = torch.eq(labels[indices_pairs[:, 0]], labels[indices_pairs[:, 1]])
        relations = relations.type(torch.float) * 2 - 1
        positive_indices = torch.where(relations == 1)[0]
        relations = relations[positive_indices]
        indices_pairs = indices_pairs[positive_indices, :]
        print(indices_pairs.shape)
        A_matrix = torch.sparse.FloatTensor(indices_pairs.T, relations, torch.Size([n_samples, n_samples]))
        return A_matrix

# labels = torch.tensor([1,2,1,1,10,3])
# A = create_big_A_matrix(labels=labels, num_links=10, only_positives=True)
# indices = A._indices().T
# values = A._values()



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





class CIFAR100(Dataset):
    def __init__(self, proportion = 1) -> None:
        self.proportion = proportion
        self.data, self.Ids = self.load_data()

    def __getitem__(self, item):
        return self.data[item,:], self.Ids[item]
    def __len__(self):
        return len(self.Ids)

    def load_data(self):
        data_train_dict = self.unpickle('CIFAR100/train')
        data_test_dict = self.unpickle('CIFAR100/test')
        meta_dict = self.unpickle('CIFAR100/meta')

        data_train = data_train_dict[b'data']
        train_Ids = np.array(data_train_dict[b'coarse_labels'])
        data_test = data_test_dict[b'data']
        test_Ids = np.array(data_test_dict[b'coarse_labels'])

        classes_names = [label.decode('utf-8') for label in meta_dict[b'coarse_label_names']]

        data = np.vstack((data_train, data_test))
        data = data.reshape(len(data), 3, 32, 32)
        data = torch.from_numpy(data)
        Ids = np.hstack((train_Ids, test_Ids))
        Ids = torch.from_numpy(Ids)
        data, Ids = self.keep_part_of_dataset(data, Ids, self.proportion)
        return data, Ids

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    @staticmethod
    def keep_part_of_dataset(data: torch.Tensor, labels: torch.Tensor, proportion: int=1):
        assert (proportion <= 1) and (proportion > 0), 'proportion must be set to a number between 0 and 1'
        n_samples = labels.numel()
        n_samples_keep = int(n_samples * proportion)
        numbers = random.sample(range(n_samples), n_samples_keep)
        return data[numbers], labels[numbers]

#dataset = CIFAR100(proportion=1/6)



class LinkedDataset(Dataset):
    '''
    contains the images that are Linked or can not link
    '''
    def __init__(self, cifardataset: CIFAR10, num_links: int = 1000):
        self.data, self.A_matrix, self.labels_subset, self.picked_indices = random_links2label(cifardataset.data, cifardataset.Ids, num_links=num_links)
        self.linked_indices = self.find_linked_indices()    # this list contains all the related images. e.g at [0] we have all the indices of A_matrix which are not 0
        #self.related_images, self.relations, self.knowledge_list = self.organize_images()
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


class UnifiedDataset(Dataset):
    def __init__(self, data: torch.Tensor, Ids: torch.Tensor, neighbor_indices: torch.Tensor,
                 neighbors_distances: torch.Tensor=None, proportion_links: int=0):
        self.data = data
        self.Ids = Ids
        self.proportion_links = proportion_links
        self.neighbor_indices = neighbor_indices
        self.neighbors_distances = neighbors_distances
        if neighbors_distances != None:
            self.neighbor_weights = F.softmax(self.neighbors_distances, dim=1)
        elif neighbors_distances == None:
            self.neighbor_weights = torch.ones(neighbor_indices.size(), dtype=torch.float)
        
        self.all_neighbors_indices, self.all_weights = self.consider_links(only_correct=False)
        self.check_neighbors_function()
        print('DONE WITH PREPROCESSING')

    def __getitem__(self, item):
        image = self.data[item]
        image_neighbors_indices = self.all_neighbors_indices[item]
        image_neighbors_weights = self.all_weights[item]

        n_neighbors = image_neighbors_indices.numel()
        random_index_index = torch.randint(0, n_neighbors, (1,)).item()
        random_neighbor_index = image_neighbors_indices[random_index_index]
        random_weight = image_neighbors_weights[random_index_index]
        random_neighbor = self.data[random_neighbor_index]
        return image, random_neighbor, random_weight, self.Ids[item]
    
    def __len__(self):
        return self.Ids.numel()


    def check_neighbors_function(self):
        n_samples = self.Ids.numel()
        for i in range(0, n_samples):
            label = self.Ids[i]
            neighbor_indices = self.all_neighbors_indices[i]
            neighbor_labels = self.Ids[neighbor_indices]
            weights = self.all_weights[i]
            x = torch.where(weights == -1)[0]
            if x.numel() == 0:
                continue
            assert torch.all(neighbor_labels[x] != label).item(), 'there is a problem with the code'


    def consider_links(self, only_correct: bool = False):
        all_neighbors = []
        all_weights = []

        n_samples = self.Ids.numel()
        if self.proportion_links != 0:
            num_additions = 0
            num_corrections = 0
            A_matrix = create_big_A_matrix(self.Ids, proportion_links=self.proportion_links, only_positives=False) # THIS IS A SPARSE TENSOR
            linked_indices = A_matrix._indices().T
            values = A_matrix._values()
            if only_correct == False:
                for i in range(0, n_samples):
                    neighbors = self.neighbor_indices[i,:]
                    weights = self.neighbor_weights[i,:]
                    ii = torch.where(linked_indices[:, 0] == i)[0]
                    if ii.numel() != 0:
                        linked_neighbors = linked_indices[ii, 1]
                        v = values[ii]
                        for j, z in enumerate(linked_neighbors):
                            if torch.isin(z, neighbors).item():
                                weights[torch.where(neighbors == z)[0]] = v[j]
                                num_corrections += 1
                            else:
                                neighbors = torch.cat((neighbors, z.unsqueeze(0)), dim=0)
                                weights = torch.cat((weights, v[j].unsqueeze(0)), dim=0)
                                num_additions += 1
                    all_neighbors.append(neighbors)
                    all_weights.append(weights)
                print('THE NUMBER OF ADDITIONS IS: ', num_additions)
                print('THE NUMBER OF CORRECTIONS IS: ', num_corrections)
                return all_neighbors, all_weights
            
            elif only_correct == True:
                num_corrections = 0
                num_additions = 0
                for i in range(0, n_samples):
                    neighbors = self.neighbor_indices[i,:]
                    weights = self.neighbor_weights[i,:]
                    ii = torch.where(linked_indices[:, 0] == i)[0]
                   # print(ii)
                    if ii.numel() != 0:
                        linked_neighbors = linked_indices[ii, 1]
                        v = values[ii]
                        for j, z in enumerate(linked_neighbors):
                            if torch.isin(z, neighbors).item() and (v[j] == -1).item():
                                num_corrections += 1
                                t = torch.where(neighbors == z)[0]
                                neighbors = torch.cat([neighbors[0:t], neighbors[t+1:]])
                                weights = torch.cat([weights[0:t], weights[t+1:]])
                            elif (torch.isin(z, neighbors)).item() == False and (v[j] == 1).item():
                                num_additions += 1
                                neighbors = torch.cat([neighbors, z.unsqueeze(0)], dim=0)
                                weights = torch.cat([weights, v[j].unsqueeze(0)], dim=0)
                    all_neighbors.append(neighbors)
                    all_weights.append(weights)
                print('THE NUMBER OF CORRECTIONS IS: ', num_corrections)
                print('THE NUMBER OF ADDITIONS IS: ', num_additions)
                return all_neighbors, all_weights
            
        elif self.proportion_links == 0:
            all_neighbors = [row for row in self.neighbor_indices]
            all_weights = [row for row in self.neighbor_weights]
            return all_neighbors, all_weights
    @staticmethod
    def max_tensor_length(tensor_list: list) -> int:
        max_length = 0


        for tensor in tensor_list:
            if len(tensor) > max_length:
                max_length = len(max_length)
        
        return max_length
    
    @staticmethod
    def minimum_tensor_length(tensor_list: list) -> int:
        min_length = 10000000

        for tensor in tensor_list:
            if len(tensor) < min_length:
                min_length = len(tensor)
        return min_length

















