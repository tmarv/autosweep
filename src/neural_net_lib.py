#!/usr/bin/env python3
# Tim Marvel
import torch
import torch.nn as nn


# TODO rename this correctly
class ThreeByThreeSig(nn.Module):

    def __init__(self):
        super(ThreeByThreeSig, self).__init__()
        self.layer1 = nn.Linear(9, 512)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.active2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 128)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x


class FiveByFiveSig(nn.Module):

    def __init__(self):
        super(FiveByFiveSig, self).__init__()
        self.layer1 = nn.Linear(25, 512)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.active2 = nn.ReLU()
        self.layer3 = nn.Linear(512, 128)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x
