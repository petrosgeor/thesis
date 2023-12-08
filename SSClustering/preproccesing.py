import pickle
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def plot_image(image: np.ndarray) -> None:
    plt.figure()
    plt.imshow(image)
    plt.show()


'''
meta_file = 'cifar-10/batches.meta'
meta_data = unpickle(meta_file)
'''

dataset_train = CIFAR10(root='C:/Users/Peter/PycharmProjects/Thesis/SSClustering', train=True, transform=None, download=False)
dataset_test = CIFAR10(root='C:/Users/Peter/PycharmProjects/Thesis/SSClustering', train=False, transform=None, download=True)

data = np.concatenate((dataset_train.data, dataset_test.data), axis=0)
Ids = np.hstack((dataset_train.targets, dataset_test.targets))
class2Id = dataset_train.class_to_idx

np.save('CIFAR10/data', data)
np.save('CIFAR10/Ids', Ids)

with open('CIFAR10/class2Id.pkl', 'wb') as file:
    pickle.dump(class2Id, file)









