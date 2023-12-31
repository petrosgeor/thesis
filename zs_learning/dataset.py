import pickle
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
from torchvision.io import read_image
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
from torch.nn import functional as F
from os.path import join

'''class AwA2dataset_features(Dataset):
    def __init__(self, path = 'AwA2-features/Animals_with_Attributes2/Features/ResNet101/'):
        self.path = path
        self.data, self.Ids, self.image_names = self.get_files()
        self.Ids = self.Ids - 1
        self.class2Id, self.Id2class = self.get_class2Id()
        #self.embedding_dictionary = self.get_classCLIPembeddings()      # a dictionary with keys the classes Id's and as values the corresponding CLIP embedding
        self.num_classes = len(set(self.image_names))

        self.test_classes_names = ['leopard', 'pig', 'hippopotamus', 'seal', 'persian+cat', 'chimpanzee', 'rat',
                                   'humpback+whale', 'giant+panda', 'raccoon']
        self.train_classes_names = list(set(list(self.class2Id.keys())) - set(self.test_classes_names))
        self.known_Ids = torch.tensor([self.class2Id[name] for name in self.train_classes_names]) - 1
        self.zs_Ids = torch.tensor([self.class2Id[name] for name in self.test_classes_names]) - 1
        self.masked_Ids = self.make_masked_Ids()
    def __getitem__(self, item):
        X = self.data[item, :]
        label_id = self.Ids[item]
        name = self.image_names[item]
        #label_embedding = self.embedding_dictionary[label_id]
        return X, label_id, name



    def __len__(self):
        return len(self.image_labels)

    def get_files(self):
        image_features1 = np.loadtxt(self.path + 'features_train'+'.txt')
        image_features2 = np.loadtxt(self.path + 'features_test.txt')
        image_features = np.vstack((image_features1, image_features2))
        image_labels1 = np.loadtxt(self.path + 'image_labels_train.txt').astype(int)
        image_labels2 = np.loadtxt(self.path + 'image_labels_test.txt').astype(int)
        image_labels = np.hstack((image_labels1, image_labels2))

        with open(self.path + 'image_names_train' + '.pkl', 'rb') as file:
            image_names1 = pickle.load(file)
        with open(self.path + 'image_names_test' + '.pkl', 'rb') as file:
            image_names2 = pickle.load(file)
        image_names = image_names1 + image_names2
        return torch.from_numpy(image_features), torch.from_numpy(image_labels), image_names


    def get_class2Id(self):
        with open(self.path + 'class2Id.pkl', 'rb') as file:
            class2Id = pickle.load(file)
        Id2class = {v: k for k, v in class2Id.items()}
        return class2Id, Id2class

    def make_masked_Ids(self):          # returns a list where if an instance has an unknown Id, then it makes it -1
        masked_Ids = self.Ids.clone()
        for i, id in enumerate(masked_Ids):
            if torch.isin(id, self.zs_Ids).item():
                masked_Ids[i] = -1
        return masked_Ids

'''

'''
class AwA2dataset_features:
    def __init__(self, path = 'AwA2-features/Animals_with_Attributes2/Features/ResNet101'):
        self.path = path
        self.data, self.Ids, self.filenames = self.get_files()

        self.test_classes_names = ['leopard', 'pig', 'hippopotamus', 'seal', 'persian+cat', 'chimpanzee', 'rat',
                                   'humpback+whale', 'giant+panda', 'raccoon']
        self.neighbor_indices = find_indices_of_closest_embeddings(embedings=F.normalize(self.data, dim=1), n_neighbors=20)

    def get_files(self):
        data = np.loadtxt(os.path.join(self.path, 'AwA2-features.txt'))
        Ids = np.loadtxt(os.path.join(self.path, 'AwA2-labels.txt'), dtype=np.int32)

        with open(os.path.join(self.path, 'AwA2-filenames.txt'), 'r') as file:
            lines = file.readlines()
        
        data = torch.from_numpy(data)
        Ids = torch.from_numpy(Ids)
        filenames = [line.strip() for line in lines]
        return data, Ids, filenames
'''



def plot_image_from_tensor(tensor):
    numpy_image = tensor.permute(1,2,0).numpy()
    plt.imshow(numpy_image)
    plt.axis('off')
    plt.show()


class AwA2dataset(Dataset):
    def __init__(self):
        self.path = set_AwA2_dataset_path()
        self.filenames, self.neighbor_indices = self.get_filenames_neighbors()
        self.data, self.Ids, self.class2Id, self.Id2class = self.load_data()
        self.known_Ids, self.zs_Ids = self.find_known_zs_Ids()
        self.masked_Ids = self.make_masked_Ids()

        self.all_neighbors_indices = self.correct_neighbors()   # this is a list containing the neighbors of each image

        self.Id2attribute = self.load_attributes()
        print('done creating the dataset')

    def __getitem__(self, item):
        related_indices = self.all_neighbors_indices[item]
        n_neighbors = related_indices.numel()
        random_index_index = torch.randint(0, n_neighbors, (1,)).item()
        random_index = related_indices[random_index_index]
        return self.data[item, :], self.data[random_index, :], self.Ids[item], self.masked_Ids[item]

    def __len__(self):
        return self.Ids.numel()

    def correct_neighbors(self):
        all_neighbors_indices = []
        for i, id in enumerate(self.Ids):
            if torch.isin(id, self.known_Ids).item():
                x = torch.where(self.Ids == id)[0]
                all_neighbors_indices.append(x)
            else:
                all_neighbors_indices.append(self.neighbor_indices[i,:])
        return all_neighbors_indices

    def load_data(self):
        classes_path = os.path.join(self.path, 'classes.txt')
        with open(classes_path, 'r') as file:
            lines = file.readlines()
        Id2class = {}
        for line in lines:
            parts = line.strip().split('\t')
            key = int(parts[0]) - 1
            value = parts[1]
            Id2class[key] = value
        
        class2Id = {value: key for key, value in Id2class.items()}

        transform = v2.Compose([
            v2.Resize((64, 64), interpolation=Image.BICUBIC, antialias=True),
            v2.ToTensor()
        ])
        images_path = os.path.join(self.path, 'JPEGImages')
        data = []
        Ids = []
        for filename in self.filenames:
            c = self.get_first_part(filename)
            c_path = join(images_path, c)

            image = read_image(join(c_path, filename))  # this is a torch tensor
            if image.shape[0] != 3:
                    image = image.repeat(3,1,1)
            image = transform(image)
            data.append(image.unsqueeze(0))
            Ids.append(class2Id[c])

        data = torch.cat(data, dim=0)
        Ids = torch.tensor(Ids)
        return data, Ids, class2Id, Id2class
    
    def find_known_zs_Ids(self):
        classes_path = os.path.join(self.path, 'trainclasses.txt')
        string_list = []

        with open(classes_path, 'r') as file:
            string_list = [line.strip() for line in file]
        known_Ids = [self.class2Id[i] for i in string_list]
        known_Ids = torch.tensor(known_Ids)

        classes_path = os.path.join(self.path, 'testclasses.txt')
        string_list = []
        
        with open(classes_path, 'r') as file:
            string_list = [line.strip() for line in file]
        zs_Ids = [self.class2Id[i] for i in string_list]
        zs_Ids = torch.tensor(zs_Ids)

        return known_Ids, zs_Ids


    def make_masked_Ids(self):
        masked_Ids = self.Ids.clone()
        for i, id in enumerate(masked_Ids):
            if torch.isin(id, self.zs_Ids).item():
                masked_Ids[i] = -1
        return masked_Ids

    @staticmethod
    def get_first_part(string):
        parts = string.split('_')
        return parts[0]

    @staticmethod
    def get_filenames_neighbors():
        path = 'AwA2-features/Animals_with_Attributes2/Features/ResNet101'
        data = np.loadtxt(os.path.join(path, 'AwA2-features.txt'))
        Ids = np.loadtxt(os.path.join(path, 'AwA2-labels.txt'), dtype=np.int32)

        with open(os.path.join(path, 'AwA2-filenames.txt'), 'r') as file:
            lines = file.readlines()
        
        data = torch.from_numpy(data)
        Ids = torch.from_numpy(Ids)
        filenames = [line.strip() for line in lines]

        neighbor_indices = find_indices_of_closest_embeddings(embedings=F.normalize(data, dim=1), n_neighbors=20)
        return filenames, neighbor_indices

    def load_attributes(self) -> dict:
        path = set_AwA2_dataset_path()
        path = join(path, 'predicate-matrix-continuous.txt')
        attributes_matrix = np.loadtxt(path)
        Id2attribute = {}
        for i in self.Id2class.keys():
            Id2attribute[i] = torch.from_numpy(attributes_matrix[i, :])
        return Id2attribute

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


def plot_histogram_NN(Ids: torch.Tensor, indices: torch.Tensor):
    correct = []
    for i, id in enumerate(Ids):
        neighbor_indices = indices[i, :]
        n_correct = torch.where(Ids[neighbor_indices] == id)[0].numel()
        correct.append(n_correct)
    plt.figure(figsize=(8, 6))  # Set the figure size (optional)
    plt.hist(correct, bins=indices.shape[1], color='skyblue', edgecolor='black')  
    plt.title('Histogram of 20 Distinct Values') 
    plt.xlabel('Values') 
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def correct_neighbors_mean(Ids: torch.Tensor, indices: torch.Tensor):
    correct = []
    for i, id in enumerate(Ids):
        neighbor_indices = indices[i, :]
        n_correct = torch.where(Ids[neighbor_indices] == id)[0].numel()
        correct.append(n_correct)
    mean = np.mean(correct)
    print(f"the mean of the correct nearest neighbors is: {mean:.2f}")


def plot_histogram_NN_zs_Ids(Ids: torch.Tensor, masked_Ids: torch.Tensor, indices: torch.Tensor):
    correct = []
    zs_indices = torch.where(masked_Ids == -1)[0]
    for i in zs_indices:
        neighbor_indices = indices[i.item(), :]
        n_correct = torch.where(Ids[neighbor_indices] == Ids[i.item()])[0].numel()
        correct.append(n_correct)
    plt.figure(figsize=(8, 6))  # Set the figure size (optional)
    plt.hist(correct, bins=indices.shape[1], color='skyblue', edgecolor='black')  
    plt.title('Histogram of 20 Distinct Values') 
    plt.xlabel('Values') 
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.show()



#dataset = AwA2dataset()

#dataset = AwA2dataset()
# plot_histogram_NN(Ids=dataset.Ids, indices=indices)
