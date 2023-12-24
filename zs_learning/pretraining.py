import torch
import torchvision
import torch.nn.functional as F
from lightly.data import LightlyDataset
from lightly.transforms import MoCoV2Transform
import lightly.transforms as transforms
from lightly.loss import NTXentLoss
import os
from utils import *
from models import *
import pytorch_lightning as pl
import torch.nn as nn
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum
)
import copy



device = 'cuda'
# Set the CUDA_VISIBLE_DEVICES environment variable to the desired GPU ID
gpu_id = input("Enter the GPU ID to be used (e.g., 0, 1, 2, ...): ")
# 

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


num_workers = 2
batch_size = 128
seed = 1
max_epochs = 800
input_size = 128
memory_bank_size = 2000


path_to_data = set_AwA2_dataset_path() + 'JPEGImages/'

transform = MoCoV2Transform(input_size=input_size)

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

class ContrastiveModel(pl.LightningModule):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        self.criterion = NTXentLoss(temperature=0.1, memory_bank_size=memory_bank_size)

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.contrastive_head_momentum = copy.deepcopy(self.contrastive_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.contrastive_head_momentum)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.contrastive_head, self.contrastive_head_momentum, 0.99)

        q = self.contrastive_head(self.backbone(x_q))
        q = F.normalize(q, dim=1)
        #print(q)

        k, shuffle = batch_shuffle(x_k)

        k = self.contrastive_head_momentum(self.backbone_momentum(k))
        k = F.normalize(k, dim=1)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]
    




# model = SimCLRModel()
#iter(dataloader_train_simclr)

model = ContrastiveModel(backbone=resnet18())
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
trainer.fit(model, dataloader_train_simclr)

torch.save(model.state_dict(), 'NeuralNets/moco_pretrained_allmodel.pth')
torch.save(model.backbone.state_dict(), 'NeuralNets/moco_pretrained_backbone.pth')




