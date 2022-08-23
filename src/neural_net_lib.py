#!/usr/bin/env python3
# Tim Marvel
import torch
import torch.nn as nn


# TODO rename this correctly
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


class FiveByFiveConv(nn.Module):

    def __init__(self):
        super(FiveByFiveConv, self).__init__()
        self.layer1 = nn.Conv2d(1, 128, 5)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 128)
        self.active2 = nn.ReLU()

        '''
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
        '''

    def forward(self, x):
        print(x)
        x = self.layer1(x)
        print(x)
        x = self.active1(x)
        print(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        print("after x.size(0), -1)")
        print(x)
        exit()
        '''
        x = self.layer2(x)
        #x = self.batchNorm1(x)
        x = self.active2(x)
        x = self.layer3(x)
        #x = self.batchNorm2(x)
        x = self.active3(x)
        x = self.layer4(x)
        '''
        return x


