import torch
import numpy as np
import faiss




x = np.random.randn(10000, 100)
x = np.array([[1,2],
             [10,20],
             [-1,-2,],
             [100,200]])


index = faiss.IndexFlatIP(x.shape[1])

index.add(x)
distances, indices = index.search(x, 2)


