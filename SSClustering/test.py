import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding




x = torch.randn((100, 100))
print(torch.where(x >= 0.5))


