import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import *
from dataset import *
import pandas as pd

def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'


    X_embedded = TSNE(n_components=2).fit_transform(resnet_embeddings)
    unique_labels = np.unique(labels)

    num_colors = len(unique_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    for i, color in zip(unique_labels, colors):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=[color], label=f'Label {i}')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Scatter Plot of Samples by Label')
    plt.show()



def VisualizeResnetFeatures(mode: str) -> None:
    assert (mode == 'pretraining') | (mode == 'EM') | (mode == 'FixMatch'), 'the mode should either be pretraining or EM or FixMatch'
    identity_aug = Identity_Augmentation()
    weak_aug = Weak_Augmentation()
    strong_aug = Strong_Augmentation()
    dataset = CIFAR100(identity_transform=identity_aug, weak_transform=weak_aug, strong_transform=strong_aug)
    
    num_classes = len(dataset.zs_Ids) + len(dataset.zs_Ids)

    if mode == 'pretraining':
        specific_path = 'Neural_Networks\PretrainingResults\PretrainedModelStateDict.pth'
        net, _, _ = LoadPretrainedNeuralNet()
        dataset.pretrain = True

    elif mode == 'EM':
        specific_path = 'Neural_Networks/EMresults/EMtrainedNetStateDict.pth'
        net = LoadEMtrainedNeuralNet(num_classes=num_classes)
        dataset.pretrain = False

    elif mode == 'FixMatch':
        specific_path = 'Neural_Networks/FixMatchResults/FixMatchNetStateDict.pth'
        #TODO
    
    env_path = find_environment()
    path = env_path + specific_path
    net.to(device)

    dataloader = CustomDataloader(known_Ids=dataset.known_Ids,
                                     zs_Ids=dataset.zs_Ids,
                                     dataset=dataset,
                                     batch_size=300,
                                     shuffle=False
                                     )
    embeddings = []
    labels = []
    if dataset.pretrain == True:
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(dataloader):
                X_batch, _, _, batch_Ids = data
                X_batch = X_batch.to(device)

                batch_embeddings = net.forward_r(X_batch)
                embeddings.append(batch_embeddings.cpu())
                labels.append(batch_Ids)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        VisualizeWithTSNE(embeddings.numpy(), labels.numpy())



#VisualizeResnetFeatures(mode='pretraining')

def LogRegrAccuracy():
    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/Neural_Networks/PretrainingResults/'
    else:
        path = 'Neural_Networks/PretrainingResults/'
    
    with open(path + 'LogRegrAccuracy.pkl', 'rb') as file:
        loaded_list = pickle.load(file)
    
    epochs = np.arange(1, len(loaded_list) + 1)
    plt.plot(epochs, loaded_list)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()

    
def plotNMI():
    if running_on_colab() == True:
        path = '/content/drive/MyDrive/data/Neural_Networks/EMresults/'
    else:
        path = 'Neural_Networks/EMresults/'
    
    data = pd.read_csv(path + 'clustering_results.csv')
    NMI = data['NMI'].values
    n = np.arange(1, len(NMI) + 1)

    plt.plot(n, NMI)
    plt.xlabel('Epoch')
    plt.ylabel('NMI')
    plt.title('NMI vs Epoch')
    plt.grid(True)
    plt.show()






