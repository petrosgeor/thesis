import torch
import numpy as np
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = np.random.randn(1000, 100)
x_gpu = torch.tensor(x).to(device)

# x = np.array([[1,2],
#              [10,20],
#              [-1,-2,],
#              [100,200]])


index = faiss.IndexFlatIP(x_gpu.shape[1])

index = faiss.index_cpu_to_all_gpus(index)
index.add(x_gpu)
distances, indices = index.search(x, 2)


