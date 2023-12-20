import torch
from torch.nn import functional as F
import torch.optim as optim
from dataset import CIFAR100, plot_image_from_tensor
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from utils import *
from models import *
import numpy as np


device = 'cuda'

torch.manual_seed(42)






















