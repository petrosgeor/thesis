import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS, LocallyLinearEmbedding
from scipy.special import binom


x = np.array([1,2,3,4])
y = np.random.permutation(x)







# mean = 35
# std = 5


# x = np.floor(np.random.normal(mean, std, 60000))
# plt.hist(x, bins=60)
# plt.xlabel('Values')
# plt.ylabel('Frequence')
# plt.title('Histogram Example')
# plt.grid(True)
# plt.show()


# x = np.arange(0, 50)
# n = 50

# def f(x, n):
#     return (x * (x-1))/(n * (n-1))
    

# y = f(x, n)
# plt.scatter(x, y)
# plt.xlabel('number of neighbors with the same label')
# plt.ylabel('fraction of correct links')
# plt.grid(True)
# plt.show()

# x = np.random.randn(60000, 128)
# mds = MDS()
# mds.fit(x)
# print(mds.dissimilarity_matrix_.shape)





