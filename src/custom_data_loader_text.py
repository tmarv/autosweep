#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import src.tools
from src import reward_manager, tools
#import reward_manager, tools


class CustomDatasetFromTextFiles3(Dataset):

    def __init__(self):
        self.positive_percent = 0.5
        self.filename3 = tools.get_text_file_names()[0]
        print(self.filename3)
        self.dataPoints = pd.read_csv(self.filename3, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        # print(self.dataPoints.head)
        self.dataPoints = np.array(self.dataPoints).astype(np.float32)

        self.rewards = pd.read_csv(self.filename3, header=None, usecols=[9])
        # print(self.rewards.head)
        self.rewards = np.array(self.rewards).astype(np.float32)
        self.dataset_size = len(self.dataPoints)

    def __getitem__(self, index):
        return self.dataPoints[index], self.rewards[index]

    def __len__(self):
        return self.dataset_size


class CustomDatasetFromTextFiles5(Dataset):

    def __init__(self):
        self.positive_percent = 0.5
        self.filename5 = tools.get_text_file_names()[1]
        print(self.filename5)
        self.dataPoints = pd.read_csv(self.filename5, header=None,
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        # print(self.dataPoints.head)
        self.dataPoints = np.array(self.dataPoints).astype(np.float32)

        self.rewards = pd.read_csv(self.filename5, header=None, usecols=[25])
        # print(self.rewards.head)
        self.rewards = np.array(self.rewards).astype(np.float32)
        self.dataset_size = len(self.dataPoints)

    def __getitem__(self, index):
        return self.dataPoints[index], self.rewards[index]

    def __len__(self):
        return self.dataset_size

'''
dataSet = CustomDatasetFromTextFiles5()
print(dataSet.__len__())
print("----")
print(dataSet.__getitem__(0))
'''