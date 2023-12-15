import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE







def VisualizeWithTSNE(resnet_embeddings: np.ndarray, labels: np.ndarray) -> None:
    assert (type(resnet_embeddings) == np.ndarray) and (type(labels) == np.ndarray), 'input should be numpy arrays not tensors or lists'
    plt.switch_backend('Agg')
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
    #plt.show()
    plt.savefig('NeuralNets/plots/first_plot.png')
    plt.switch_backend('TkAgg')
