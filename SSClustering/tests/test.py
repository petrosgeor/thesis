import torch
import numpy as np
import random


num_links = 5000

labels = torch.randint(0, 10, size=(60000,))
n_samples = labels.numel()
indices = torch.arange(0, n_samples, step=1).tolist()

random_pairs = random.sample(indices, k=2*num_links)
indices_pairs = [torch.tensor([random_pairs[i], random_pairs[i+1]]) for i in range(0, len(random_pairs), 2)]
indices_pairs = torch.stack(indices_pairs, dim=0)

indices_pairs = torch.cat((indices_pairs, indices_pairs[:, [1,0]]), dim=0)

relations = torch.eq(labels[indices_pairs[:,0]], labels[indices_pairs[:, 1]])
relations = relations.type(torch.float) * 2 - 1

A_matrix = torch.sparse.FloatTensor(indices_pairs.T, relations, torch.Size([60000,60000]))