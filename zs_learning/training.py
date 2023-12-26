from utils import *
from evaluation import *
from augmentations import *
from dataset import *
from models import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def scan_training(num_epochs: int=200, num_classes:int = 50):
    dataset = AwA2dataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

    clusternet = ClusteringModel(backbone=resnet18(), nclusters=num_classes, nheads=1)
    clusternet.to(device)

    



