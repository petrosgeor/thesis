import torch
import numpy as np
from dataset import *
from utils import *
from augmentations import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from evaluation import *
import torch.optim as optim

class SemanticNetwork(nn.Module):
    def __init__(self):
        super(SemanticNetwork, self).__init__()

        self.sequential = nn.Sequential(
                                        nn.Linear(85, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
        )
    
    def forward(self, x):
        return F.normalize(self.sequential(x), dim=1)



def plot_image_from_tensor(tensor):
    numpy_image = tensor.permute(1,2,0).numpy()
    plt.imshow(numpy_image)
    plt.axis('off')
    #plt.show()
    plt.savefig('plots/aaaaa')


device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

x = resnet18()
resnet = x['backbone']
resnet.load_state_dict(torch.load('NeuralNets/backbone_AwA2.pth'))

clusternet = ClusteringModel(backbone={'backbone':resnet, 'dim': 512}, nclusters=50)
clusternet.cluster_head.load_state_dict(torch.load('NeuralNets/cluster_head_AwA2.pth'))
dataset = AwA2dataset()
id_aug = Identity_Augmentation()

dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
clusternet.to(device)
predictions = []
embeddings = []
labels = []
clusternet.eval()
with torch.no_grad():
    for i, (X_batch, _, labels_batch, _) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        X_batch = id_aug(X_batch)
        

        x = clusternet.forward(X_batch, forward_pass='return_all')
        embeddings.append(x['features'].cpu())
        batch_probs = F.softmax(x['output'][0], dim=1)

        batch_predictions = torch.argmax(batch_probs, dim=1)

        predictions.append(batch_predictions.cpu())
        labels.append(labels_batch)


predictions = torch.cat(predictions, dim=0)
labels = torch.cat(labels, dim=0)
predictions2labels = torch.from_numpy(get_y_preds(y_true=labels.numpy(), cluster_assignments=predictions.numpy(), n_clusters=50))
predictions2labels = predictions2labels.to(torch.int)
embeddings = torch.cat(embeddings, dim=0)
embeddings = F.normalize(embeddings, dim=1)

Ids = torch.unique(predictions, sorted=True)
means = []
for i, id in enumerate(Ids):
    indices = torch.where(predictions == id)[0]
    m = embeddings[indices, :].mean(dim=0).unsqueeze(0)
    means.append(m)

means = torch.cat(means, dim=0)
Y = np.loadtxt(set_AwA2_dataset_path() + '/predicate-matrix-continuous.txt')
Y = torch.from_numpy(Y)
# similarity_matrix = torch.matmul(means, embeddings.T)
# best_examples = torch.topk(similarity_matrix, k=10, dim=1)[1]

known_Ids = dataset.known_Ids
zs_Ids = dataset.zs_Ids

d = {}
for i, id in enumerate(predictions):
    d[id.item()] = predictions2labels[i].item()

keys = torch.tensor(list(d.keys()))
values = torch.tensor(list(d.values()))

x = torch.where(torch.isin(values, known_Ids))[0]
visual_c = keys[x]      # indices of clusters which correspond to a known class
semantic_c = values[x]  # indices of classes which are mapped to known clusters

t = torch.arange(0, 50)

visual_n = torch.masked_select(t, ~torch.isin(t, visual_c))
semantic_n = torch.masked_select(t, ~torch.isin(t, semantic_c))


semantic_net = SemanticNetwork()
optimizer = optim.Adam(semantic_net.parameters(), lr=10**(-6), weight_decay=10**(-2))
num_epochs = 30000
gradient_steps = 3

for epoch in range(0, num_epochs):
    with torch.no_grad():
        Y_embedded = semantic_net(Y.float())
        similarities = 1 - torch.matmul(Y_embedded[semantic_n,:], means[visual_n,:].T)
        P = find_permutation_matrix(similarities)
    
    for j in range(0, gradient_steps):
        Y_embedded = semantic_net(Y.float())
        sup_similarities = 1 - (Y_embedded[semantic_c,:] * means[visual_c,:]).sum(dim=1)
        sup_loss = torch.mean(sup_similarities)

        zs_similarities = 1 - torch.matmul(Y_embedded[semantic_n,:], means[visual_n,:].T)
        zs_loss = torch.mean(P * zs_similarities)

        print(sup_loss.item(), zs_loss.item())
        total_loss = 2*sup_loss + zs_loss * (epoch/num_epochs)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


Y_embedded = Y_embedded.detach()


similarities = torch.matmul(means, Y_embedded.T)
x = torch.argmax(similarities, dim=1)
preds_transformed = predictions.clone()
for i, t in enumerate(x):
    ii = torch.where(predictions == i)[0]
    preds_transformed[ii] = t

known_labels = torch.masked_select(labels, torch.isin(labels, known_Ids))
known_predictions = torch.masked_select(preds_transformed, torch.isin(labels, known_Ids))
print(torch.where(known_labels == known_predictions)[0].numel()/known_labels.numel() * 100)


zs_labels = torch.masked_select(labels, torch.isin(labels, zs_Ids))
zs_predictions = torch.masked_select(preds_transformed, torch.isin(labels, zs_Ids))

print(torch.where(zs_labels == zs_predictions)[0].numel()/zs_labels.numel()*100)


# similarities = torch.matmul(embeddings, Y_embedded.T)

# x = torch.argmax(similarities, dim=1)

# known_labels = torch.masked_select(labels, torch.isin(labels, known_Ids))
# known_predictions = torch.masked_select(x, torch.isin(labels, known_Ids))

# print(torch.where(known_labels == known_predictions)[0].numel()/known_labels.numel() * 100)

# zs_labels = torch.masked_select(labels, torch.isin(labels, zs_Ids))
# zs_predictions = torch.masked_select(x, torch.isin(labels, zs_Ids))

# print(torch.where(zs_labels == zs_predictions)[0].numel()/zs_labels.numel()*100)





















def plot_histogram_backbone_NN():
    #clusternet = initialize_clustering_net(n_classes=50, nheads=1)
    id_aug = Identity_Augmentation()
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=60, shuffle=False)
    embeddings = []
    Ids = []
    clusternet.to(device)
    clusternet.eval()
    with torch.no_grad():
        for i, (X_batch, _, labels_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            X_batch = id_aug(X_batch)
            embeddings_batch = clusternet.forward(X_batch, forward_pass='backbone')
            embeddings.append(embeddings_batch.cpu())
            Ids.append(labels_batch)

        embeddings = torch.cat(embeddings, dim=0)
        Ids = torch.cat(Ids, dim=0)
        indices = find_indices_of_closest_embeddings(F.normalize(embeddings, dim=1), n_neighbors=20)
        #Ids = dataset.Ids
        correct = []
        for i, id in enumerate(Ids):
            neighbor_indices = indices[i, :]    
            n_correct = torch.where(Ids[neighbor_indices] == id)[0].numel()
            correct.append(n_correct)
        
        plt.figure(figsize=(8, 6))  # Set the figure size (optional)
        plt.hist(correct, bins=20, color='skyblue', edgecolor='black')  
        plt.title('Histogram of 20 Distinct Values') 
        plt.xlabel('Values') 
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.5)
        plt.show()


#plot_histogram_backbone_NN()










