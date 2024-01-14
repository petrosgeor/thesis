import torch
import numpy as np
from dataset import *
from utils import *
from augmentations import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from evaluation import *
import torch.optim as optim
from sklearn.manifold import TSNE
from utils import *
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope

###     CHECK RESULTS OF MODEL TRAINED END TO END WITH SCAN     #### 

device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


class SemanticNetwork(nn.Module):
    def __init__(self):
        super(SemanticNetwork, self).__init__()

        self.sequential = nn.Sequential(
                                        nn.Linear(85, 85),
                                        nn.ReLU(),
                                        nn.Linear(85, 2),
                                        #nn.Sigmoid(),
                                        #nn.Linear(85,2)
        )
    
    def forward(self, x):
        #return F.normalize(self.sequential(x), dim=1)
        return self.sequential(x)


def generate_new_correspondances(visual_samples: torch.Tensor, semantic_samples: torch.Tensor, num_new: int=20) -> torch.Tensor:
    n_samples = visual_samples.shape[0]
    indices = torch.randint(low=0, high=n_samples, size=(num_new, 2))
    new_visual_samples = []
    new_semantic_samples = []
    for row in indices:
        x1 = visual_samples[row[0]]
        x2 = visual_samples[row[1]]
        x = 1/2*(x1 + x2)
        new_visual_samples.append(x)
        y1 = semantic_samples[row[0]]
        y2 = semantic_samples[row[1]]
        y = 1/2*(y1 + y2)
        new_semantic_samples.append(y)
    new_visual_samples = torch.stack(new_visual_samples, dim=0)
    new_semantic_samples = torch.stack(new_semantic_samples, dim=0)
    return new_visual_samples, new_semantic_samples


def plot_image_from_tensor(tensor):
    numpy_image = tensor.permute(1,2,0).numpy()
    plt.imshow(numpy_image)
    plt.axis('off')
    #plt.show()
    plt.savefig('plots/aaaaa')


x = give_resnet18()
resnet = x['backbone']
resnet.load_state_dict(torch.load('NeuralNets/backbone_AwA2.pth'))

clusternet = ClusteringModel(backbone={'backbone':resnet, 'dim': 512}, nclusters=50)
clusternet.cluster_head.load_state_dict(torch.load('NeuralNets/cluster_head_AwA2.pth'))
dataset = AwA2dataset(training=False, size_resize=64)
id_aug = Identity_Augmentation()

dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
clusternet.to(device)

predictions = []
embeddings = []
labels = []

clusternet.eval()
with torch.no_grad():
    for i, (X_batch, labels_batch, _) in enumerate(dataloader):
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

embeddings = torch.cat(embeddings, dim=0)
embeddings = F.normalize(embeddings, dim=1)


tsne = TSNE(n_components=2, n_jobs=3)
embeddings_transformed = tsne.fit_transform(embeddings.numpy())
embeddings_transformed = torch.from_numpy(embeddings_transformed)

known_Ids = dataset.known_Ids
zs_Ids = dataset.zs_Ids
known_indices = torch.where(torch.isin(labels, known_Ids))[0]
zs_indices = torch.where(torch.isin(labels, zs_Ids))[0]

known_means = []

for i in known_Ids:
    indices = torch.where(labels == i)[0]
    m = embeddings_transformed[indices, :].mean(dim=0)
    known_means.append(m)

known_means = torch.stack(known_means, dim=0)
kmeans = KMeans(n_clusters=10)
envelope = EllipticEnvelope(contamination=0.1)
envelope.fit(embeddings_transformed[zs_indices, :].numpy())
outliers = envelope.predict(embeddings_transformed[zs_indices, :].numpy())


kmeans.fit(embeddings_transformed[zs_indices, :].numpy()[outliers == 1])
zs_means = torch.from_numpy(kmeans.cluster_centers_)

Y = np.loadtxt(set_AwA2_dataset_path() + '/predicate-matrix-continuous.txt')
Y = torch.from_numpy(Y)
Y = Y.to(device)

semantic_net = SemanticNetwork()
semantic_net.to(device)
optimizer = optim.Adam(semantic_net.parameters(), lr=10**(-4), weight_decay=10**(-2))
num_epochs = 5000
gradient_steps = 3
pairwise_distance = torch.nn.PairwiseDistance(p=2)

known_means = known_means.to(device)
zs_means = zs_means.to(device)


for epoch in range(0, num_epochs):
    with torch.no_grad():
        Y_embedded_zs = semantic_net(Y[zs_Ids, :].float())
        similarities_zs = torch.cdist(Y_embedded_zs, zs_means)
        P = find_permutation_matrix(similarities_zs.cpu()).to(device)
    for j in range(0, gradient_steps):

        Y_embedded = semantic_net(Y.float())
        sup_similarities = pairwise_distance(Y_embedded[known_Ids], known_means)
        sup_loss = torch.mean(sup_similarities)

        zs_similarities = torch.cdist(Y_embedded[zs_Ids, :], zs_means)
        zs_loss = torch.mean(P * zs_similarities)

        #loss = sup_loss + (epoch/num_epochs)*zs_loss
        loss = sup_loss + zs_loss

        print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Y_embedded = Y_embedded.detach().cpu()


similarities = torch.cdist(embeddings_transformed, Y_embedded)

x = torch.argmin(similarities, dim=1)
known_labels = torch.masked_select(labels, torch.isin(labels, known_Ids))
known_predictions = torch.masked_select(x, torch.isin(labels, known_Ids))
print(torch.where(known_labels == known_predictions)[0].numel()/known_labels.numel() * 100)

zs_labels = torch.masked_select(labels, torch.isin(labels, zs_Ids))
zs_predictions = torch.masked_select(x, torch.isin(labels, zs_Ids))

print(torch.where(zs_labels == zs_predictions)[0].numel()/zs_labels.numel()*100)


zs_embeddings = embeddings_transformed[zs_indices, :]
similarities = torch.cdist(zs_embeddings, Y_embedded)
similarities.index_fill_(dim=1, index=known_Ids, value=torch.inf)

preds = torch.argmin(similarities, dim=1)
print(torch.where(zs_labels == preds)[0].numel()/zs_labels.numel() * 100)



# visualize_embeddings(embeddings_transformed[zs_indices].numpy(), labels[zs_indices].numpy(), means=Y_embedded[zs_Ids].numpy(),
#                      Ids = zs_Ids.numpy(), path2save='plots/aaa.png')

# visualize_embeddings(embeddings_transformed.numpy(), labels.numpy(), means=Y_embedded.numpy(), Ids=np.arange(0,50), path2save='plots/bbb.png')

def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray, 
                      means:np.ndarray, zs_Ids: np.ndarray, path2save: str='plots/aaa.png', n_cpus: int=2) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'

    tsne = TSNE(n_components=2, n_jobs=n_cpus)
    X_embedded = tsne.fit_transform(resnet_embeddings)
    unique_labels = np.unique(labels)

    #num_colors = len(unique_labels)
    #colors = plt.cm.tab40(np.linspace(0, 1, num_colors))
    if len(unique_labels) == 40:
        colors = plt.cm.tab20.colors[:20] + plt.cm.tab20b.colors[:20]
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]
    elif len(unique_labels) == 50:
        colors = colors = (plt.cm.tab20.colors[:20] + plt.cm.tab20b.colors[:20] + plt.cm.tab20c.colors[:10])
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]
    elif len(unique_labels) == 10:
        colors = plt.cm.tab10.colors[:10]
        hex_colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3]) for color in colors]

    plt.figure()
    for i, color in zip(unique_labels, hex_colors):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=[color], label=f'Label {i}', s=10, alpha=0.5)

    if means != None:
        for i, (x, y) in enumerate(means):
            plt.scatter(x, y, s=50)
            plt.text(x, y, str(zs_Ids[i]), fontsize=9, ha='right')
    #plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Samples by Label')
    plt.savefig(path2save)





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