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


x = np.arange(0, 50)
n = 50

def f(x, n):
    return (x * (x-1))/(n * (n-1))
    

y = f(x, n)
plt.scatter(x, y)
plt.xlabel('number of neighbors with the same label')
plt.ylabel('fraction of correct links')
plt.grid(True)
plt.show()


