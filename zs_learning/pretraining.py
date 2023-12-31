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
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
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
batch_size = 256
seed = 1
max_epochs = 800
input_size = 64
memory_bank_size = 2000


path_to_data = set_AwA2_dataset_path() + '/JPEGImages/'

transform = MoCoV2Transform(input_size=input_size,
                            gaussian_blur=0.0,
                            )

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

class MocoModel(pl.LightningModule):
    def __init__(self):
        super(MocoModel, self).__init__()
        
        resnet = ResNetGenerator('resnet-18', 1)
        self.backbone = self.backbone = nn.Sequential(
                                        *list(resnet.children())[:-1],
                                        nn.AdaptiveAvgPool2d(1),
                                    )

        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(temperature=0.1, memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        if batch_idx == 1:
            torch.save(self.backbone.state_dict(), 'NeuralNets/moco_pretrained_backbone.pth')

        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
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
    

# model = MocoModel()
# trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
# trainer.fit(model, dataloader_train_simclr)

# torch.save(model.state_dict(), 'NeuralNets/moco_pretrained_allmodel.pth')
# torch.save(model.backbone.state_dict(), 'NeuralNets/moco_pretrained_backbone.pth')


# resnet = ResNetGenerator('resnet-18', 1)
# backbone = backbone = nn.Sequential(
#                                         *list(resnet.children())[:-1],
#                                         nn.AdaptiveAvgPool2d(1),
#                                     )



