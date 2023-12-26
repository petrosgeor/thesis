import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from PIL import Image
import random
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


np.random.seed(42)
random.seed(42)
num_zs_classes = 10  # number of zero shot classes
N = 20              # the number of classes to keep from CIFAR100 DATASET



def plot_numpy_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def split_classes(Ids: np.array):    # This function accepts the numpy array with all labels and returns the train classes and
    unique_Ids = np.unique(Ids)
    zs_classes = np.random.choice(unique_Ids, num_zs_classes, replace=False)
    not_zs_classe_indices = np.where(~np.isin(unique_Ids, zs_classes))[0]
    seen_classes = unique_Ids[not_zs_classe_indices]
    return seen_classes, zs_classes



def plot_image_from_tensor(tensor):
    numpy_image = tensor.permute(1,2,0).numpy()
    plt.imshow(numpy_image)
    plt.axis('off')
    plt.show()


class CIFAR100(Dataset):
    def __init__(self):
        self.data, self.Ids, self.classes = self.load_data()    # both data and Ids are numpy arrays

        self.Id2class, self.class2Id = self.make_dict_correspondace()
        self.known_Ids, self.zs_Ids = split_classes(self.Ids)
        self.known_indices, self.zs_indices = self.find_known_zs_indices(self.known_Ids, self.zs_Ids, self.Ids)

        self.masked_Ids = self.make_masked_Ids()

        self.data = torch.from_numpy(self.data)
        self.Ids = torch.from_numpy(self.Ids)
        self.known_Ids = torch.from_numpy(self.known_Ids)
        self.zs_Ids = torch.from_numpy(self.zs_Ids)
        self.masked_Ids = torch.from_numpy(self.masked_Ids)
        self.known_indices = torch.from_numpy(self.known_indices)
        self.zs_indices = torch.from_numpy(self.zs_indices)


    def load_data(self):

        def make_classes_list(Ids, classes):
            classes_list = []
            for i in range(0, len(Ids)):
                id = Ids[i]
                classes_list.append(classes[id])
            return classes_list


        data_train_dict = self.unpickle('CIFAR100/train')
        data_test_dict = self.unpickle('CIFAR100/test')
        meta_dict = self.unpickle('CIFAR100/meta')

        data_train = data_train_dict[b'data']
        train_Ids = np.array(data_train_dict[b'coarse_labels'])
        data_test = data_test_dict[b'data']
        test_Ids = np.array(data_test_dict[b'coarse_labels'])

        classes_names = [label.decode('utf-8') for label in meta_dict[b'coarse_label_names']]

        data = np.vstack((data_train, data_test))
        data = data.reshape(len(data),3,32,32)

        Ids = np.hstack((train_Ids, test_Ids))
        return data, Ids, make_classes_list(Ids, classes_names)


    def make_masked_Ids(self):          # returns a list where if an instance has an unknown Id, then it makes it -1
        if type(self.Ids) == np.ndarray:
            masked_Ids = self.Ids.copy()
        elif type(self.Ids == torch.Tensor):
            masked_Ids = self.Ids.clone()
        for i, id in enumerate(masked_Ids):
            if id not in self.known_Ids:
                masked_Ids[i] = -1
        return masked_Ids

    def make_dict_correspondace(self):
        x = []
        for i in range(0, len(self.classes)):
            dummy = [self.Ids[i], self.classes[i]]
            x.append(dummy)

        Id2class = {x[i][0]: x[i][1] for i in range(0, len(x))}
        class2Id = {v: k for k, v in Id2class.items()}
        return Id2class, class2Id


    def __getitem__(self, item):
        return self.data[item,:], self.Ids[item], self.masked_Ids[item]


    def __len__(self):
        return len(self.data)


    @staticmethod
    def find_known_zs_indices(known_Ids, zs_Ids, Ids):
        known_indices = np.where(np.isin(Ids, known_Ids))[0]
        zs_indices = np.where(~np.isin(Ids, known_Ids))[0]
        return known_indices, zs_indices

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict



class SCANDATASET(Dataset):
    def __init__(self, data: torch.Tensor, Ids: torch.Tensor, masked_Ids: torch.Tensor, all_neighbors_indices: list):
        self.data = data
        self.Ids = Ids
        self.masked_Ids = masked_Ids
        self.all_neighbors_indices = all_neighbors_indices

    def __getitem__(self, item):
        related_indices = self.all_neighbors_indices[item]
        n_neighbors = related_indices.numel()
        random_index_index = torch.randint(0, n_neighbors, (1,)).item()
        random_index = related_indices[random_index_index]
        return self.data[item, :], self.data[random_index, :], self.Ids[item], self.masked_Ids[item]
    
    def __len__(self):
        return self.Ids.numel()
