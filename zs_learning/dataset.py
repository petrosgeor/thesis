import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def create_mapping_dicts(numbers):

    # Sort the input numbers
    sorted_numbers = sorted(numbers)

    # Create a dictionary to map input numbers to values between 0 and 12
    Id2index = {num: i for i, num in enumerate(sorted_numbers)}

    # Create a dictionary to map values between 0 and 12 to the corresponding input numbers
    index2Id = {i: num for i, num in enumerate(sorted_numbers)}

    return Id2index, index2Id

def keep_part_of_dictionary(dictionary, keys2keep):
    to_keep = {}
    for i in range(0, len(keys2keep)):
        key = keys2keep[i]
        to_keep[key] = dictionary[key]
    return to_keep



class AwA2dataset(Dataset):
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
    # def get_classCLIPembeddings(self):
    #     model_id = "openai/clip-vit-base-patch32"
    #     prompt = 'a photo of a '
    #     tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    #     processor = CLIPProcessor.from_pretrained(model_id)
    #     model = CLIPModel.from_pretrained(model_id)
    #     embedding_dictionary = {}
    #     for i in self.Id2class.keys():
    #         s = self.Id2class[i].replace('+',' ')
    #         phrase = prompt + s
    #         token = tokenizer(phrase, return_tensors='pt')
    #         class_embedding = model.get_text_features(**token).detach().numpy()
    #         embedding_dictionary[i] = class_embedding
    #     return embedding_dictionary



def find_indices_of_closest_embeddings(embedings: torch.Tensor, n_neighbors: int = 20) -> torch.Tensor:
    D = torch.matmul(embedings, embedings.T)
    indices = torch.topk(D, k=n_neighbors, dim=1)[1]
    return indices



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




# dataset = AwA2dataset()
# zs_indices = torch.where(dataset.masked_Ids == -1)[0]
# indices = find_indices_of_closest_embeddings(dataset.data)
# plot_histogram_NN_zs_Ids(Ids=dataset.Ids, masked_Ids=dataset.masked_Ids, indices=indices)
