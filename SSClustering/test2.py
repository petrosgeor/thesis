import torch
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import Dataset, DataLoader



class Dataset1(Dataset):
    def __init__(self):
        self.data = torch.arange(10)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.numel()



class Dataset2(Dataset):
    def __init__(self):
        self.data = torch.arange(-5,0)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.numel()



dataset1 = Dataset1()
dataset2 = Dataset2()

dataloader1 = DataLoader(dataset1, batch_size=3, shuffle=False)
dataloader2 = DataLoader(dataset2, batch_size=2, shuffle=False)

for epoch in range(2):
    dl_iterator = iter(dataloader2)

    for i, X in enumerate(dataloader1):
        try:
            Y = next(dl_iterator)
        except StopIteration:
            dl_iterator = iter(dataloader2)
            Y = next(dl_iterator)
        
        print('batch from X is: ', X)
        print('batch from Y is', Y)








