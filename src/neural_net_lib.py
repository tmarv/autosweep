#!/usr/bin/env python3
# Tim Marvel
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO rename this correctly
# TODO check which works better: ReLU vs LeakyReLU
class ThreeByThreeSig(nn.Module):

    def __init__(self):
        super(ThreeByThreeSig, self).__init__()
        self.layer1 = nn.Linear(9, 256)
        #self.batchNorm1 = nn.BatchNorm1d(512)
        #self.active1 = nn.ReLU()
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, 256)
        #self.batchNorm1 = nn.BatchNorm1d(512)
        self.active2 = nn.LeakyReLU()
        #self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(256, 64)
        #self.batchNorm2 = nn.BatchNorm1d(128)
        self.active3 = nn.LeakyReLU()
        #self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        #x = self.batchNorm1(x)
        x = self.active1(x)
        x = self.layer2(x)
        #x = self.batchNorm1(x)
        x = self.active2(x)
        x = self.layer3(x)
        #x = self.batchNorm2(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x

class ThreeByThreeCluster(nn.Module):

    def __init__(self):
        super(ThreeByThreeCluster, self).__init__()
        self.layer1 = nn.Linear(9, 128)
        #self.batchNorm1 = nn.BatchNorm1d(512)
        #self.active1 = nn.ReLU()
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(128, 128)
        #self.batchNorm1 = nn.BatchNorm1d(512)
        self.active2 = nn.LeakyReLU()
        #self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, 128)
        #self.batchNorm2 = nn.BatchNorm1d(128)
        self.active3 = nn.LeakyReLU()
        #self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        #x = self.batchNorm1(x)
        x = self.active1(x)
        x = self.layer2(x)
        #x = self.batchNorm1(x)
        x = self.active2(x)
        x = self.layer3(x)
        #x = self.batchNorm2(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x

class FiveByFiveSig(nn.Module):

    def __init__(self):
        super(FiveByFiveSig, self).__init__()
        self.layer1 = nn.Linear(25, 1024)
        #self.active1 = nn.ReLU()
        self.active1 = nn.LeakyReLU()
        #self.batchNorm1 = nn.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 1024)
        #self.active2 = nn.ReLU()
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(1024, 128)
        #self.batchNorm2 = nn.BatchNorm1d(128)
        #self.active3 = nn.ReLU()
        self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(128, 1)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        #x = self.batchNorm1(x)
        x = self.active1(x)
        x = self.layer2(x)
        #x = self.batchNorm1(x)
        x = self.active2(x)
        x = self.layer3(x)
        #x = self.batchNorm2(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x

class ThreeByThreeAug(nn.Module):

    def __init__(self):
        super(ThreeByThreeAug, self).__init__()
        self.layer1 = nn.Linear(108, 128)
        #self.active1 = nn.ReLU()
        self.active1 = nn.LeakyReLU()
        self.batchNorm1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(128, 128)
        #self.active2 = nn.ReLU()
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, 128)
        self.batchNorm2 = nn.BatchNorm1d(128)
        #self.active3 = nn.ReLU()
        self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(128, 1)


    def forward(self, x):
        #x = torch.flatten(x, 1)
        x = self.layer1(x)
        #x = self.batchNorm1(x)
        x = self.active1(x)
        x = self.layer2(x)
        #x = self.batchNorm1(x)
        x = self.active2(x)
        x = self.layer3(x)
        #x = self.batchNorm2(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x


class ThreeByThreeConv(nn.Module):
    def __init__(self):
        super(ThreeByThreeConv, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.active2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x


class FiveByFiveConv(nn.Module):
    def __init__(self):
        super(FiveByFiveConv, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv_layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.activeRelu = nn.ReLU()
        self.fc_bn_256 = nn.BatchNorm1d(256)
        self.fc_layer2 = nn.Linear(512, 256)
        self.fc_layer3 = nn.Linear(256, 256)
        self.fc_layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv_layer2(x))
        x = self.conv1_bn(x)
        #x = nn.BatchNorm2d(x)
        # x = self.activeRelu(x)
        #x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer2(x)
        x = self.activeRelu(x)
        x = self.fc_bn_256(x)
        x = self.fc_layer3(x)
        x = self.activeRelu(x)
        x = self.fc_bn_256(x)
        x = self.fc_layer4(x)
        return x

class FiveByFiveConvCluster(nn.Module):
    def __init__(self):
        super(FiveByFiveConvCluster, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv_layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activeRelu = nn.ReLU()
        self.fc_bn_256 = nn.BatchNorm1d(256)
        self.fc_layer2 = nn.Linear(512, 256)
        self.fc_layer3 = nn.Linear(256, 256)
        self.fc_layer4 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        # x = self.activeRelu(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv_layer2(x))
        x = self.conv1_bn(x)
        # x = self.activeRelu(x)
        # print(x)
        # x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer2(x)
        x = self.activeRelu(x)
        x = self.fc_bn_256(x)
        x = self.fc_layer3(x)
        x = self.activeRelu(x)
        x = self.fc_bn_256(x)
        x = self.fc_layer4(x)
        return x