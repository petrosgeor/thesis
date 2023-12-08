import torch
import numpy as np
import torch.nn as nn

class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes=0):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 1),  # Modified stride to 1 for the first layer
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),  # Modified stride to 1
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),  # Modified stride to 1
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),  # Modified stride to 1
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            #conv_dw(512, 1024, 1),  # Modified stride to 1
            #conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)  # Global pooling for any input size
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        return x


model = MobileNetV1(ch_in=3)
x = torch.randn(1,3,32,32)
