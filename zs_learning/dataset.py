import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import random

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
        self.image_features, self.image_labels, self.image_names = self.get_files()
        self.class2Id, self.Id2class = self.get_class2Id()
        self.embedding_dictionary = self.get_classCLIPembeddings()      # a dictionary with keys the classes Id's and as values the corresponding CLIP embedding
        self.num_classes = len(set(self.image_names))

        self.test_classes_names = ['leopard', 'pig', 'hippopotamus', 'seal', 'persian+cat', 'chimpanzee', 'rat',
                                   'humpback+whale', 'giant+panda', 'raccoon']
        self.train_classes_names = list(set(list(self.class2Id.keys())) - set(self.test_classes_names))
        self.train_classes_Ids = [self.class2Id[name] for name in self.train_classes_names]
        self.test_classes_Ids = [self.class2Id[name] for name in self.test_classes_names]

    def __getitem__(self, item):
        X = self.image_features[item, :]
        label_id = self.image_labels[item]
        name = self.image_names[item]
        label_embedding = self.embedding_dictionary[label_id]
        return X, label_id, name, label_embedding



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
        return image_features, image_labels, image_names


    def get_class2Id(self):
        with open(self.path + 'class2Id.pkl', 'rb') as file:
            class2Id = pickle.load(file)
        Id2class = {v: k for k, v in class2Id.items()}
        return class2Id, Id2class

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
