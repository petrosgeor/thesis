import torch
import numpy as np
import random


num_links = 5000

c = torch.randint(0, 10, size=(10, 2))

linked_indices = torch.randint(0, 10, size=(10000, 2))


#equality_matrix = torch.all(c.unsqueeze(1) == linked_indices.unsqueeze(0), dim=2)


for i in range(0, c.shape[0]):
    print((torch.eq(c[i, :], linked_indices)).all(dim=1).shape)
