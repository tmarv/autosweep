#!/usr/bin/env python3
# Tim Marvel
import torch
import torch.nn as nn
import torch.nn.functional as F


class SevenBySeven1ConvLayerXLeakyReLUSigmoidEnd(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(SevenBySeven1ConvLayerXLeakyReLUSigmoidEnd, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=7, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


# net used for 7 by 7 input grid and 2 convolutional layer
# the impact of the second convolutional layer significantly improves the performance
class SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU()
            )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=sz, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU()
            )    

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=9*sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.convolutional_layer(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = torch.flatten(x, 1)
        #print(x.size())
        x = self.linear_layer(x)
        return x



# net used for 7 by 7 input grid and 2 convolutional layer
# the impact of the second convolutional layer significantly improves the performance
class SevenBySeven2ConvLayerXLeakyReLUSigmoidEndV2(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(SevenBySeven2ConvLayerXLeakyReLUSigmoidEndV2, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=5, stride=1),
            nn.LeakyReLU()
            )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=sz, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU()
            )    

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1),
            nn.Sigmoid())            

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


 # can be removed?


class ThreeByThree1ConvLayerXBatchNorm(nn.Module):
    def __init__(self, sz, dp):
        super(ThreeByThree1ConvLayerXBatchNorm, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.BatchNorm1d(sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.BatchNorm1d(sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


class ThreeByThree1ConvLayerX(nn.Module):
    def __init__(self, sz, dp):
        super(ThreeByThree1ConvLayerX, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


class SevenBySeven1ConvLayerXLeakyReLU(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(SevenBySeven1ConvLayerXLeakyReLU, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=7, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


class SevenBySeven1ConvLayerXSigmoid(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(SevenBySeven1ConvLayerXSigmoid, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=7, stride=1),
            nn.Sigmoid())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.Sigmoid(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.Sigmoid(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


class FiveByFive1ConvLayerX(nn.Module):
    def __init__(self, sz, dp):
        super(FiveByFive1ConvLayerX, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=5, stride=1),
            nn.LeakyReLU()
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


# simple fully connected network
class ThreeByThreeSig(nn.Module):

    def __init__(self):
        super(ThreeByThreeSig, self).__init__()
        self.layer1 = nn.Linear(9, 256)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, 256)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(256, 64)
        self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(64, 1)

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


# simple fully connected layer
# this net underperforms use convolution if possible
class FiveByFiveSig(nn.Module):

    def __init__(self):
        super(FiveByFiveSig, self).__init__()
        self.layer1 = nn.Linear(25, 1024)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(1024, 1024)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(1024, 128)
        self.active3 = nn.LeakyReLU()
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


class FiveByFiveConv(nn.Module):
    def __init__(self):
        super(FiveByFiveConv, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv_layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.activeRelu1 = nn.ReLU()
        self.fc_bn_256_1 = nn.BatchNorm1d(256)
        self.activeRelu2 = nn.ReLU()
        self.fc_bn_256_2 = nn.BatchNorm1d(256)
        self.fc_layer2 = nn.Linear(512, 256)
        self.fc_layer3 = nn.Linear(256, 256)
        self.fc_layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv_layer2(x))
        x = self.conv2_bn(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer2(x)
        x = self.activeRelu1(x)
        x = self.fc_bn_256_1(x)
        x = self.fc_layer3(x)
        x = self.activeRelu2(x)
        x = self.fc_bn_256_2(x)
        x = self.fc_layer4(x)
        return x