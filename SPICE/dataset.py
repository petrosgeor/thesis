import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from PIL import Image
from augmentations import Identity_Augmentation, Weak_Augmentation, Strong_Augmentation
from utils import running_on_colab
from utils import LoadEMtrainedNeuralNet
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


np.random.seed(42)
num_zs_classes = 5  # number of zero shot classes
N = 20              # the number of classes to keep from CIFAR100 DATASET

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    def __init__(self, pretrain = False):
        self.data_path = self.find_environment()
        self.pretrain = pretrain
        self.data_path_train = self.data_path + 'train'
        self.data_path_test = self.data_path + 'test'
        self.data, self.Ids, self.classes = self.load_data()


        self.Id2class, self.class2Id = self.make_dict_correspondace()
        self.known_Ids, self.zs_Ids = split_classes(self.Ids)
        self.known_indices, self.zs_indices = self.find_known_zs_indices(self.known_Ids, self.zs_Ids, self.Ids)

        self.masked_Ids = self.make_masked_Ids()


    def load_data(self):

        def make_classes_list(Ids, classes):
            classes_list = []
            for i in range(0, len(Ids)):
                id = Ids[i]
                classes_list.append(classes[id])
            return classes_list


        data_train_dict = unpickle(self.data_path_train)
        data_test_dict = unpickle(self.data_path_test)
        meta_dict = unpickle(self.data_path + 'meta')

        data_train = data_train_dict[b'data']
        train_Ids = np.array(data_train_dict[b'coarse_label_names'])
        data_test = data_test_dict[b'data']
        test_Ids = np.array(data_test_dict[b'coarse_label_names'])

        classes_names = [label.decode('utf-8') for label in meta_dict[b'coarse_label_names']]

        data = np.vstack((data_train, data_test))
        data = data.reshape(len(data),3,32,32)

        Ids = np.hstack((train_Ids, test_Ids))
        return data, Ids, make_classes_list(Ids, classes_names)


    def make_masked_Ids(self):          # returns a list where if an instance has an unknown Id, then it makes it -1
        masked_Ids = self.Ids.copy()
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
        if self.pretrain == True:                               # if we are at the pretraining stage then we need two strong augmentations of the same image to compare
            x_numpy = self.data[item]
            image = Image.fromarray(x_numpy)                    # convert the numpy array to a PIL image
            identity_image = self.identity_transform(image)
            strong1_image = self.strong_transform(image)        # First strong augmentation of the image
            strong2_image = self.strong_transform(image)        # Second strong augmentation of the image
            return identity_image, strong1_image, strong2_image, self.Ids[item]

        else:
            x_numpy = self.data[item]
            image = Image.fromarray(x_numpy)
            identity_image = self.identity_transform(image)
            weak_image = self.weak_transform(image)
            strong_image = self.strong_transform(image)
            return identity_image, weak_image, strong_image, self.Ids[item], self.masked_Ids[item]


    def __len__(self):
        return len(self.data)


    @staticmethod
    def find_known_zs_indices(known_Ids, zs_Ids, Ids):
        known_indices = np.where(np.isin(Ids, known_Ids))[0]
        zs_indices = np.where(~np.isin(Ids, known_Ids))[0]
        return known_indices, zs_indices

    @staticmethod
    def find_environment():
        if running_on_colab() == True:
            path = '/content/drive/MyDrive/data/cifar-100/'
        else:
            path = 'cifar-100/'
        return path



'''identity_aug = Identity_Augmentation()
weak_aug = Weak_Augmentation()
strong_aug = Strong_Augmentation()

dataset = CIFAR100(identity_transform=identity_aug, weak_transform=weak_aug, strong_transform=strong_aug, )

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
'''
'''Ids = []
for i, data in enumerate(dataloader):
    _, _, _, Ids_batch, _ = data
    Ids.append(Ids_batch)
Ids = torch.cat(Ids, dim=0)
'''
#net = LoadEMtrainedNeuralNet(num_classes=20)

