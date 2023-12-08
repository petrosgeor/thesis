import torch
import numpy as np
import matplotlib.pyplot as plt

'''
x = torch.randn(10, 100)
A = torch.randn(10, 20, 100)
'''

x = torch.tensor([[1,2,3],[10,20,30]])

A = torch.tensor([[[1,2,3],
                   [-71,-2,-3],
                   [100, 100, 100]],
                  [[-1, -2, -3],
                   [1, 22, 3],
                   [35, 64, 1234]]])

x_transposed = x.unsqueeze(1)

result = torch.bmm(A, x_transposed.transpose(1,2))

