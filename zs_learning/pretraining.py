import torch
import torchvision
import lightly
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform
import lightly.transforms as transforms
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
import os
from utils import *
import pytorch_lightning as pl
import torch.nn as nn

device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
# 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


num_workers = 1
batch_size = 256
seed = 1
max_epochs = 20
input_size = 64
num_ftrs = 32



path_to_data = set_AwA2_dataset_path() + 'JPEGImages/'

transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5)

# We create a torchvision transformation for embedding the dataset after
# training
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=transforms.utils.IMAGENET_NORMALIZE["mean"],
            std=transforms.utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)
dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

# model = SimCLRModel()
# trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
iter(dataloader_train_simclr)
# trainer.fit(model, dataloader_train_simclr)

# train.py

# Your code here
# ...

if __name__ == "__main__":
    for batch in dataloader_train_simclr:
        break
