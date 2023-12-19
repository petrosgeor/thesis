import torch
from torch.nn import functional as F
import torch.optim as optim
from randaugment import *
from dataset import CIFAR100, plot_image_from_tensor
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from utils import *
from models import *
import numpy as np


device = set_device()
device = 'cuda'

torch.manual_seed(42)


def contrastive_training(dataloader, num_epochs=1000, t_contrastive=0.5):
    try:
        net, current_epoch, current_lr = LoadPretrainedNeuralNet()
        print('----------The training is continued\n-----------------')
        print('The epoch we left off was: ',current_epoch)
    except:
        resnet, hidden_dim = get_resnet('resnet18')
        net = Network(resnet=resnet, hidden_dim=hidden_dim, feature_dim=128, class_num=100)
        current_epoch = 0
        current_lr = 10**(-4)
        print('No model found, the training starts from scratch\n')

    net.float()
    net.to(device)
    criterion = InfoNCELoss(temperature=t_contrastive)
    optimizer = optim.Adam(net.parameters(), lr=current_lr, weight_decay=10**(-4))

    for ep in range(0, num_epochs - current_epoch):
        net.to(device)
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            _, X_batch1, X_batch2, _ = data
            X_batch1 = X_batch1.to(device)
            X_batch2 = X_batch2.to(device)

            z1 = net(X_batch1)
            z2 = net(X_batch2)

            loss = criterion(z1, z2)
            epoch_loss += loss.detach().item()
            '''if i % 10 == 0:
                print(loss.detach().item())'''
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        current_epoch += 1
        lr = optimizer.param_groups[0]['lr']
        if ep%2 == 0:
            #SavePretrainedNeuralNet(neural_net=net, epoch=current_epoch, current_lr=lr)
            pass
        print('the loss for epoch: ', ep,' is: ', epoch_loss)

        print(PretrainedModelEvaluation(model=net, dataloader=dataloader))


def EvaluationDatasetCreation(model, dataloader):
    with torch.no_grad():
        features = []
        labels = []
        for i, data in enumerate(dataloader):
            X_batch, _, _, Ids = data
            X_batch = X_batch.to(device)
            Ids = Ids.to(device)
            embeddings = model.forward_r(X_batch)       # ResNet features
            features.append(embeddings)
            labels.append(Ids)

        embeddings = torch.vstack(features)
        labels = torch.hstack(labels)
        known_indices = torch.where(torch.isin(labels, dataloader.known_Ids))[0]
        train_labels = labels[known_indices]
        train_embeddings = embeddings[known_indices, :]
        return train_embeddings.to('cpu'), (F.one_hot(train_labels.type(torch.int64))).to('cpu')


def PretrainedModelEvaluation(model, dataloader):
    embeddings, labels = EvaluationDatasetCreation(model, dataloader)
    n_features = embeddings.shape[1]
    one_hot_dim = labels.shape[1]

    class EvaluationDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __getitem__(self, item):
            return self.embeddings[item, :], self.labels[item, :]

        def __len__(self):
            return self.embeddings.shape[0]

    whole_dataset = EvaluationDataset(embeddings, labels)
    train_dataset, test_dataset = random_split(whole_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    accuracy = TrainLogRegrOnFeatures(train_loader, test_loader, n_features, n_classes=one_hot_dim)
    return accuracy


def TrainLogRegrOnFeatures(train_loader, test_loader, n_features, n_classes):
    LogRegrModel = LogisticRegr(n_features_in=n_features, n_classes=n_classes)
    LogRegrModel.to(device)

    optimizer = optim.Adam(LogRegrModel.parameters(), lr=10**(-3), weight_decay=10**(-4))
    earlystopping = EarlyStopping(patience=5, delta=2.)
    criterion = nn.CrossEntropyLoss()

    epoch = 1
    total_test_samples = test_loader.dataset.__len__()
    while 1 > 0:
        correct_predictions = 0
        for i, data in enumerate(train_loader):
            X_batch, y_batch = data
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predicted_probs = LogRegrModel(X_batch)
            loss = criterion(predicted_probs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                X_batch, y_batch = data
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predicted_probs = LogRegrModel(X_batch)
                predicted_classes = torch.argmax(predicted_probs, dim=1)
                true_classes = torch.argmax(y_batch, dim=1)

                correct_predictions += (predicted_classes == true_classes).sum().detach().item()

            accuracy = round(correct_predictions/total_test_samples * 100, ndigits=2)
            earlystopping(accuracy)
            if epoch % 5 == 0:
                print('the accuracy for epoch: ',epoch,' is ', accuracy)
            if earlystopping.early_stop:
                break
            epoch += 1
    return accuracy





def ExpMaxAlgorithmTraining(train_dataloader: CustomDataloader, eval_loader: CustomDataloader, num_epochs: int, M: int, K:int) -> Network:
    ''' dataloader: A custom dataloader that i have created in the utils file
        M: is the batch size
        K: is the number of clusters
    '''
    try:
        net = LoadEMtrainedNeuralNet()
        net = ChangeFinalLayerDim(net, 20)
        net = (FreezeResnet(net)).to(device)
        print('Neural network found, the training is continued')

    except:
        print('Expectation Maximization has not yet begun')
        net, _, _ = LoadPretrainedNeuralNet()
        net = ChangeFinalLayerDim(net, 20)
        net = (FreezeResnet(net)).to(device)
        
    known_Ids = torch.sort(train_dataloader.known_Ids)[0]
    zs_Ids = torch.sort(train_dataloader.zs_Ids)[0]
    known_Ids = known_Ids.to(device)
    zs_Ids = zs_Ids.to(device)
    ratio = round(M/K)

    num_classes = net.cluster_num
    all_Ids = torch.arange(0, num_classes, device=device)
    print('the number of classes is: ', num_classes)
    
    criterion = CustomCrossEntropyLoss(num_classes=known_Ids.shape[0] + zs_Ids.shape[0])    # my custom cross entropy loss built in the utils folder
    optimizer = optim.Adam(net.parameters(), lr=10**(-3))
    ClusterMetrics = ClusteringMetrics(num_classes=known_Ids.shape[0] + zs_Ids.shape[0], seen_classes=known_Ids, zs_classes=zs_Ids, saving_file=None)                # this is a class for keeping track of the accuracy of the model

    centers = torch.zeros((net.cluster_num, 512), dtype=torch.float, device=device)

    for epoch in range(0, num_epochs):
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            print(i)
            X_identity, X_weak, X_strong, _, labels = data
            labels = labels.type(torch.LongTensor)
            X_identity = X_identity.to(device)
            X_weak = X_weak.to(device)
            X_strong = X_strong.to(device)
            labels = labels.to(device)

            Embeddings = net.forward_r(X_identity)  # embeddings of the whole batch (these are going to be used in order to calculate the class centers)

            unknown_indices = torch.where(labels == -1)[0]  # finds the indices where the label is -1 (unknown)
            known_indices = torch.where(labels != -1)[0]
            X_unknown_weak = X_weak[unknown_indices, :]     # keep only the samples which we don't know the label
            Probs = net.forward_c(X_unknown_weak)[:, zs_Ids]     # keeping the probabilities only for the unknown clusters
            indices_to_keep = torch.topk(Probs, k=ratio, dim=0, largest=True, sorted=True)[1] # 1 is for keeping only the indices and not the values
            
            #print(indices_to_keep)
            one_hot_matrix = torch.zeros((labels.shape[0], num_classes), device=device)

            for j in known_Ids:
                k = torch.where(labels == j)[0]
                centers[j, :] = torch.mean(Embeddings[k, :], dim=0)

            for j, id in enumerate(zs_Ids):
                centers[id, :] = torch.mean(Embeddings[unknown_indices[indices_to_keep[:, j]], :], dim=0)

            Distances = torch.matmul(Embeddings, centers.T)
            nearest_examples = torch.topk(Distances, k=ratio, dim=0, largest=True, sorted=True)[1]
            one_hot_matrix[nearest_examples.T.flatten(), all_Ids.repeat(ratio, 1).T.flatten()] = 1

            #MAXIMIZATION STEP
            predictions = net.forward_c(X_strong)
            loss2 = kl_divergance(predictions[known_indices, :], predictions[unknown_indices, :])
            loss1 = criterion.forwardOneHot(F.softmax(predictions, dim=1), one_hot_matrix)         # In the original paiper they add another softmax function here
            loss = loss1 + loss2 * 10**(-3)
            print(loss.detach())
            epoch_loss += loss.detach().item()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        print('the training loss for epoch:', epoch,' is: ', epoch_loss)

        net.eval()
        with torch.no_grad():
            predictions = []        # predictions for all instances for one epoch
            true_labels = []        # the true labels for all instances for one epoch
            for i, data in enumerate(eval_loader):
                X_identity, _, _, labels, _ = data
                labels = labels.type(torch.LongTensor)
                X_identity = X_identity.to(device)
                labels = labels.to(device)

                probs = net.forward_c(X_identity)
                batch_predictions = torch.argmax(probs, dim=1)

                predictions.append(batch_predictions)
                true_labels.append(labels)
            
            predictions = torch.cat(predictions, dim=0) # 1d tensor containing all the predictions
            true_labels = torch.cat(true_labels, dim=0) # 1d tensor containing all the true labels

            ClusterMetrics(labels_true=true_labels.cpu(), labels_pred=predictions.cpu())
            SaveEMtrainedNeuralNet(net)
            ClusteringMetrics.plot_confusion_matrix(true_labels.cpu(), predictions.cpu())
    


def ExpMaxAlgorithmTraining1(train_dataloader: CustomDataloader, eval_loader: CustomDataloader, num_epochs: int, M: int, K: int,
                             TV_pretrained: bool=False) -> Network:
    if TV_pretrained == False:
        try:
            net = LoadEMtrainedNeuralNet(num_classes=K)
            net = (FreezeResnet(net)).to(device)
            print('Neural network found, the training is continued')

        except:
            print('Expectation Maximization has not yet begun')
            net, _, _ = LoadPretrainedNeuralNet()
            net = ChangeFinalLayerDim(net, 20)          # changes the cluster projector final layers
            net = (FreezeResnet(net)).to(device)
    elif TV_pretrained == True:
        resnet, hidden_dim = get_TVpretrained_resnet('resnet18')
        net = Network(resnet, hidden_dim, feature_dim=128, class_num=K)
        net = FreezeResnet(net).to(device)

    known_Ids = train_dataloader.known_Ids
    zs_Ids = train_dataloader.zs_Ids
    known_Ids = known_Ids.to(device)
    zs_Ids = zs_Ids.to(device)
    ratio = round(M/K)

    num_classes = net.cluster_num
    print('the number of classes is: ', num_classes)
    
    criterion = CustomCrossEntropyLoss(num_classes=known_Ids.shape[0] + zs_Ids.shape[0])    # my custom cross entropy loss built in the utils folder
    optimizer = optim.Adam(net.parameters(), lr=10**(-4))
    ClusterMetrics = ClusteringMetrics(num_classes=known_Ids.shape[0] + zs_Ids.shape[0], seen_classes=known_Ids, zs_classes=zs_Ids, saving_file=None)                # this is a class for keeping track of the accuracy of the model

    target_distribution = torch.full(size=(num_classes,), fill_value=1/num_classes).to(device)
    ClusterLoss = Cluster_KL(num_classes, target_distribution)

    for epoch in range(0, num_epochs):
        epoch_loss = 0
        net.train()
        for i, data in enumerate(train_dataloader):
            print(i)
            
            X_identity, X_weak, X_strong, _, labels = data
            labels = labels.type(torch.LongTensor)
            X_identity = X_identity.to(device)
            X_weak = X_weak.to(device)
            X_strong = X_strong.to(device)
            labels = labels.to(device)

            known_indices = torch.where(labels != -1)[0]    # indices in the batch where we know the label
            unknown_indices = torch.where(labels == -1)[0]  # indices where we do not know the label

            #supervised_loss = criterion(net.forward_c(X_strong[known_indices, :]), labels[known_indices])
            supervised_loss = criterion(net.forward_c(X_identity[known_indices, :]), labels[known_indices])     # training using only the identity transformation

            probs = net.forward_c(X_weak[unknown_indices, :])[:, zs_Ids]   # we predict only the 
            best_indices = torch.topk(probs, k=ratio, dim=0, largest=True, sorted=False)[1]

            X_strong_predictions = net.forward_c(X_strong)

            one_hot_matrix = torch.zeros((unknown_indices.shape[0], num_classes), device=device)

            one_hot_matrix[best_indices.T.flatten(), zs_Ids.repeat(ratio, 1).T.flatten()] = 1
            unsupervised_loss = criterion.forwardOneHot(X_strong_predictions[unknown_indices, :], one_hot_matrix)
            kl_loss = ClusterLoss(X_strong_predictions)

            #loss = supervised_loss + unsupervised_loss - kl_loss * 10**(-3)
            loss = supervised_loss + unsupervised_loss + kl_loss*10
            print('supervised loss is: ',supervised_loss.item(), 'unsupervised_loss is: ',unsupervised_loss.item(),' Kl loss is: ', kl_loss.item())
            epoch_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print('the training loss for epoch:', epoch,' is: ', epoch_loss)

        net.eval()
        with torch.no_grad():
            predictions = []        # predictions for all instances for one epoch
            true_labels = []        # the true labels for all instances for one epoch
            for i, data in enumerate(eval_loader):
                X_identity, _, _, labels, _ = data
                labels = labels.type(torch.LongTensor)
                X_identity = X_identity.to(device)
                labels = labels.to(device)

                probs = net.forward_c(X_identity)
                batch_predictions = torch.argmax(probs, dim=1)

                predictions.append(batch_predictions)
                true_labels.append(labels)
            
            predictions = torch.cat(predictions, dim=0) # 1d tensor containing all the predictions
            true_labels = torch.cat(true_labels, dim=0) # 1d tensor containing all the true labels

            ClusterMetrics(labels_true=true_labels.cpu(), labels_pred=predictions.cpu())
            SaveEMtrainedNeuralNet(net)
            ClusteringMetrics.plot_confusion_matrix(true_labels.cpu(), predictions.cpu())






identity_aug = Identity_Augmentation()
weak_aug = Weak_Augmentation()
strong_aug = Strong_Augmentation()
dataset = CIFAR100(identity_transform=identity_aug,strong_transform=strong_aug, weak_transform=weak_aug, pretrain=False)
'''test_subset = Subset(dataset, list(np.random.randint(0, 30000, (1000, ))))
dataloader = CustomDataloader(known_Ids=dataset.known_Ids,
                             zs_Ids=dataset.zs_Ids,
                             dataset=test_subset,
                             batch_size=300,
                             shuffle=True
                             )'''


train_dataloader = CustomDataloader(known_Ids=dataset.known_Ids,
                                     zs_Ids=dataset.zs_Ids,
                                     dataset=dataset,
                                     batch_size=300,
                                     shuffle=True
                                     )
eval_loader = CustomDataloader(known_Ids=dataset.known_Ids,
                               zs_Ids=dataset.zs_Ids,
                               dataset=dataset,
                               batch_size=300,
                               shuffle=False)


ExpMaxAlgorithmTraining1(train_dataloader=train_dataloader, eval_loader=eval_loader, num_epochs=5, M=300, K=25, TV_pretrained=True)

