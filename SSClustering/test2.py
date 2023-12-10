import torch
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import Dataset, DataLoader

from losses import losses


loss = losses.ClusterConsistencyLoss()







