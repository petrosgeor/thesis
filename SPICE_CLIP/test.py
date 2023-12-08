
from utils import *
from dataset import *
from randaugment import *
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
import torch.nn as nn




net = nn.Sequential(
    nn.Linear(100,50),
    nn.ReLU(),
    nn.Linear(50, 10)
)


network = net()

x = torch.randn(1, 100, 1)
net(x)
























