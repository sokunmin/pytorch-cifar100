"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun 

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasic(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        
    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        bottleneck_channel = int(out_channels / 4)
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_channel, 1, stride=stride),

            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, bottleneck_channel, 3, padding=1),

            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, out_channels, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64, 256, 1)
        self.stage2 = self._make_layers(block, num_block[1], 256, 512, 2)
        self.stage3 = self._make_layers(block, num_block[2], 512, 1024, 2)
        self.stage4 = self._make_layers(block, num_block[3], 1024, 2048, 2)

        self.linear = nn.Linear(2048, class_num)
    
    def _make_layers(self, block, num, in_channels, out_channels, stride):
        layers = []

        layers.append(block(in_channels, out_channels, stride))
        while num - 1:
            layers.append(block(out_channels, out_channels, 1))
            num -= 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def preactresnet18():
    return PreActResNet(PreActBasic, [2, 2, 2, 2])
    
def preactresnet34():
    return PreActResNet(PreActBasic, [3, 4, 6, 3])

def preactresnet50():
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3])

def preactresnet101():
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3])

def preactresnet152():
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3])