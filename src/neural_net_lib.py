#!/usr/bin/env python3
# Tim Marvel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeByThreeProbofchng1ConvLayer(nn.Module):
    def __init__(self):
        super(ThreeByThreeProbofchng1ConvLayer, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(256, 256)
        self.active2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.drop3(x)
        x = self.layer4(x)
        return x


class ThreeByThree1ConvLayer512(nn.Module):
    def __init__(self,drop):
        super(ThreeByThree1ConvLayer512, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(drop)
        self.layer2 = nn.Linear(512, 256)
        self.active2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(drop)
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.drop3(x)
        x = self.layer4(x)
        return x


class ThreeByThree1ConvLayer16BatchNorm(nn.Module):
    def __init__(self):
        super(ThreeByThree1ConvLayer16BatchNorm, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=16, out_features=16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(in_features=16, out_features=16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(16, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


#Sigmoid
class ThreeByThree1ConvLayerXBatchNormSigmoid(nn.Module):
    def __init__(self, sz, dp=0.0):
        super(ThreeByThree1ConvLayerXBatchNormSigmoid, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=3, stride=1),
            nn.Sigmoid())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=sz, out_features=sz),
            nn.BatchNorm1d(sz),
            nn.Sigmoid(),
            nn.Dropout(dp),
            nn.Linear(in_features=sz, out_features=sz),
            nn.BatchNorm1d(sz),
            nn.Sigmoid(),
            nn.Dropout(dp),
            nn.Linear(sz, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


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


class FiveByFive1ConvLayerX(nn.Module):
    def __init__(self, sz, dp):
        super(FiveByFive1ConvLayerX, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=sz, kernel_size=5, stride=1),
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


class FiveByFive2ConvLayerX(nn.Module):
    def __init__(self, sz, dp):
        super(FiveByFive2ConvLayerX, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=sz, kernel_size=3, stride=1),
            nn.LeakyReLU(),
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


class ThreeByThree1ConvLayer64BatchNorm(nn.Module):
    def __init__(self):
        super(ThreeByThree1ConvLayer64BatchNorm, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x

#CBRD
class ThreeByThree1ConvLayer512BatchNorm(nn.Module):
    def __init__(self):
        super(ThreeByThree1ConvLayer512BatchNorm, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            #nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            #nn.Dropout(0.25),
            nn.Linear(256, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x


class ThreeByThree1ConvLayer2048BatchNorm(nn.Module):
    def __init__(self):
        super(ThreeByThree1ConvLayer2048BatchNorm, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=3, stride=1),
            nn.LeakyReLU())

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            #nn.Dropout(0.25),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            #nn.Dropout(0.25),
            nn.Linear(256, 1))

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)
        return x

class ThreeByThree1ConvLayer2048(nn.Module):
    def __init__(self,drop):
        super(ThreeByThree1ConvLayer2048, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=2048, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(drop)
        self.layer2 = nn.Linear(2048, 256)
        self.active2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(drop)
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.drop3(x)
        x = self.layer4(x)
        return x


class ThreeByThreeCon2DDouble(nn.Module):
    def __init__(self,dropout):
        super(ThreeByThreeCon2DDouble, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(64, 64)
        self.active3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout(dropout)
        self.layer4 = nn.Linear(64, 64)
        self.active4 = nn.ReLU()
        self.drop4 = nn.Dropout(dropout)
        self.layer5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = torch.flatten(x, 1)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.drop3(x)
        x = self.layer4(x)
        x = self.active4(x)
        x = self.drop4(x)
        x = self.layer5(x)
        return x


class ThreeByThreeProbofchng1ConvLayer(nn.Module):
    def __init__(self):
        super(ThreeByThreeProbofchng1ConvLayer, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, 256)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(256, 1)
        self.active4 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        #x = self.active4(x)
        return x

class ThreeByThreeProbofchng1ConvLayerSGMD(nn.Module):
    def __init__(self):
        super(ThreeByThreeProbofchng1ConvLayerSGMD, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, 256)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(256, 256)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(256, 2)
        self.active4 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        #x = self.active4(x)
        return x


class ThreeByThreeProbofchng1ConvLayerLarger(nn.Module):
    def __init__(self):
        super(ThreeByThreeProbofchng1ConvLayerLarger, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=4096, kernel_size=3, stride=1)
        #self.layer1b = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=1)
        #self.layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=1)
        self.active1 = nn.LeakyReLU(0.1)
        #self.active1b = nn.LeakyReLU(0.3)
        self.layer2 = nn.Linear(4096, 1024)
        self.active2 = nn.LeakyReLU(0.1)
        self.layer3 = nn.Linear(1024, 512)
        self.active3 = nn.LeakyReLU(0.1)
        self.layer4 = nn.Linear(512, 1)
        self.active4 = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = torch.flatten(x, 1)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        #x = self.active4(x)
        return x


class ThreeByThreeRelu16_d2(nn.Module):
    def __init__(self):
        super(ThreeByThreeRelu16_d2, self).__init__()
        self.layer1 = nn.Linear(9, 16)
        self.active1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(16, 16)
        self.active2 = nn.LeakyReLU(0.1)
        self.layer3 = nn.Linear(16, 16)
        self.active3 = nn.LeakyReLU(0.1)
        self.layer4 = nn.Linear(16, 1)

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


class ThreeByThreeRelu32_d2(nn.Module):
    def __init__(self):
        super(ThreeByThreeRelu32_d2, self).__init__()
        self.layer1 = nn.Linear(9, 32)
        self.active1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(32, 32)
        self.active2 = nn.LeakyReLU(0.1)
        self.layer3 = nn.Linear(32, 32)
        self.active3 = nn.LeakyReLU(0.1)
        self.layer4 = nn.Linear(32, 1)

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


class ThreeByThreeRelu64_d2(nn.Module):
    def __init__(self):
        super(ThreeByThreeRelu64_d2, self).__init__()
        self.layer1 = nn.Linear(9, 64)
        self.active1 = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(64, 64)
        self.active2 = nn.LeakyReLU(0.1)
        self.layer3 = nn.Linear(64, 64)
        self.active3 = nn.LeakyReLU(0.1)
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


class ThreeByThreeRelu128_d2(nn.Module):
    def __init__(self):
        super(ThreeByThreeRelu128_d2, self).__init__()
        self.layer1 = nn.Linear(9, 128)
        self.active1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 128)
        self.active2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(128, 128)
        self.active3 = nn.LeakyReLU(0.1)
        self.drop3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.active1(x)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.drop3(x)
        x = self.layer4(x)
        return x
# TODO rename this correctly
# TODO check which works better: ReLU vs LeakyReLU
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

# TODO: what about batchnorm?
class ThreeByThreeCluster(nn.Module):

    def __init__(self):
        super(ThreeByThreeCluster, self).__init__()
        self.layer1 = nn.Linear(9, 128)
        self.active1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(128, 128)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, 128)
        self.active3 = nn.LeakyReLU()
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

class ThreeByThreeAug(nn.Module):
    def __init__(self):
        super(ThreeByThreeAug, self).__init__()
        self.layer1 = nn.Linear(108, 128)
        self.active1 = nn.LeakyReLU()
        self.batchNorm1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(128, 128)
        self.active2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, 128)
        self.batchNorm2 = nn.BatchNorm1d(128)
        self.active3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(128, 1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
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

class FiveByFiveConvCluster(nn.Module):
    def __init__(self):
        super(FiveByFiveConvCluster, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv_layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.activeRelu1 = nn.ReLU()
        self.activeRelu2 = nn.ReLU()
        self.fc_bn_256_1 = nn.BatchNorm1d(512)
        self.fc_bn_256_2 = nn.BatchNorm1d(256)
        self.fc_layer2 = nn.Linear(512, 512)
        self.fc_layer3 = nn.Linear(512, 256)
        self.fc_layer4 = nn.Linear(256, 3)

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