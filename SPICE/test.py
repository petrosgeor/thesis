
from utils import *
from dataset import *
from randaugment import *

import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")




identity_aug = Identity_Augmentation()
weak_aug = Weak_Augmentation()
strong_aug = Strong_Augmentation()
dataset = CIFAR100(identity_transform=identity_aug,strong_transform=strong_aug, weak_transform=weak_aug, pretrain=False)
dataloader = DataLoader(dataset, batch_size=300, shuffle=False)


for batch in dataloader:
    break

X_batch = batch[0]
inputs = processor(text=None, images=X_batch, return_tensors="pt")


#with torch.no_grad():
#    image_features = model(X_batch)

















