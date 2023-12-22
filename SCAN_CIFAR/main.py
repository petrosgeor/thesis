import pickle
import matplotlib.pyplot as plt
from dataset import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


d = unpickle('cifar-100/train')


data_path = CIFAR100.find_environment()
data_path_train = data_path + 'train'
data_path_test = data_path + 'test'


def load_data():

    def make_classes_list(Ids, classes):
        classes_list = []
        for i in range(0, len(Ids)):
            id = Ids[i]
            classes_list.append(classes[id])
        return classes_list


    data_train_dict = unpickle(data_path_train)
    data_test_dict = unpickle(data_path_test)
    meta_dict = unpickle(data_path + 'meta')

    data_train = data_train_dict[b'data']
    train_Ids = np.array(data_train_dict[b'coarse_labels'])
    data_test = data_test_dict[b'data']
    test_Ids = np.array(data_test_dict[b'coarse_labels'])

    classes_names = [label.decode('utf-8') for label in meta_dict[b'coarse_label_names']]

    data = np.vstack((data_train, data_test))
    data = data.reshape(len(data),3,32,32).transpose(0,2,3,1)

    Ids = np.hstack((train_Ids, test_Ids))
    return keep_part_of_dataset(data, Ids, make_classes_list(Ids, classes_names), N=N)


data, Ids, names = load_data()


