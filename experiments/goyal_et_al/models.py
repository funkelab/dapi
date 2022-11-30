import torch

class Vgg2D(torch.nn.Module):
    def __init__(
            self,
            input_size,
            input_channels,
            fmaps=12,
            downsample_factors=[(2, 2), (2, 2), (2, 2), (2, 2)],
            output_classes=6):

        self.input_size = input_size
        self.input_channels = input_channels

        super(Vgg2D, self).__init__()

        current_fmaps = input_channels
        current_size = tuple(input_size)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv2d(
                    current_fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = tuple(
                int(c/d)
                for c, d in zip(current_size, downsample_factors[i]))
            check = (
                s*d == c
                for s, d, c in zip(size, downsample_factors[i], current_size))
            assert all(check), \
                "Can not downsample %s by chosen downsample factor" % \
                (current_size,)
            current_size = size

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] *
                current_size[1] *
                current_fmaps,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                output_classes)
        ]

        self.classifier = torch.nn.Sequential(*classifier)

        print(self)
    
    def forward(self, raw):
        f = self.features(raw)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
    

import torch
from torch import nn
import math

class ResNet(nn.Module):
    def __init__(self, output_classes, input_size=(128,128), input_channels=1, downsample_factors=None):
        super(ResNet, self).__init__()
        self.in_channels = 12
        size = input_size[0]
        self.conv = nn.Conv2d(input_channels, self.in_channels, kernel_size=3,
                              padding=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()

        current_channels = self.in_channels
        self.layer1 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        size /= 2
        self.layer2 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        size /= 2
        self.layer3 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        size /= 2
        self.layer4 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        size /= 2
        size = int(math.ceil(size))

        fc = [torch.nn.Linear(current_channels*size**2, 4096),
              torch.nn.ReLU(),
              torch.nn.Dropout(),
              torch.nn.Linear(4096, 4096),
              torch.nn.ReLU(),
              torch.nn.Dropout(),
              torch.nn.Linear(4096,output_classes)]

        self.fc = torch.nn.Sequential(*fc)
        print(self)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if  (stride != 1) or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    #custom helper function 
    #to make network have same API as VGG
    def features(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
    
    def classifier(self, x):
        return  self.fc(x)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # Biases are handled by BN layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = nn.ReLU()(out)
        return out