import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding



mean = 35
std = 5


x = np.floor(np.random.normal(mean, std, 60000))
plt.hist(x, bins=60)
plt.xlabel('Values')
plt.ylabel('Frequence')
plt.title('Histogram Example')
plt.grid(True)
plt.show()


