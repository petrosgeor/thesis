import torch
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity






x = torch.tensor([[1, 2, 3, 4, 5],
                  [10, 20, 30, 40, 50]], dtype=torch.float)

y = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)

print(torch.matmul(y, y))






