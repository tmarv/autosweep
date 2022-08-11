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

    def __init__(self, is_small=False, is_clean=False, with_var=False, cluster_num=-2):
        self.filename3 = tools.get_text_file_names()[0]

        if is_small:
            self.filename3 = tools.get_text_file_names_small()[0]
        if is_clean:
            self.filename3 = tools.get_text_file_names_clean()[0]
        if with_var:
            self.filename3 = tools.get_text_file_names_var()[0]
        print(self.filename3)
        self.dataPoints = pd.read_csv(self.filename3, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.dataPoints = np.array(self.dataPoints).astype(np.float32)
        self.rewards = pd.read_csv(self.filename3, header=None, usecols=[9])
        self.rewards = np.array(self.rewards).astype(np.float32)
        # try capping rewards? calling a rewardshaping function?
        self.dataset_size = len(self.dataPoints)
        self.with_cluster = False
        #if we are training for a specific cluster
        if cluster_num>-2:
            self.clusters = np.array(pd.read_csv(self.filename3, header=None, usecols=[11]))
            self.rewards = np.array(pd.read_csv(self.filename3, header=None, usecols=[11]))
            self.with_cluster = True
            if cluster_num>-1:
                self.with_cluster = False
                clustered_data = []
                cluster_rewards = []
                for i in range (self.dataset_size):
                    if self.clusters[i] == cluster_num:
                        clustered_data.append(self.dataPoints[i])
                        cluster_rewards.append(self.rewards[i])

                self.dataPoints = np.array(clustered_data).astype(np.float32)
                self.rewards = np.array(cluster_rewards)
                self.dataset_size = len(self.dataPoints)
    def __getitem__(self, index):
        return self.dataPoints[index], self.rewards[index]


    def __len__(self):
        return self.dataset_size


class CustomDatasetFromTextFiles5(Dataset):

    def __init__(self, isSmall=False):
        self.positive_percent = 0.5
        self.filename5 = tools.get_text_file_names()[1]

        if isSmall:
            self.filename5 = tools.get_text_file_names_small()[1]

        print(self.filename5)
        self.dataPoints = pd.read_csv(self.filename5, header=None,
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        self.dataPoints = np.array(self.dataPoints).astype(np.float32)
        self.rewards = pd.read_csv(self.filename5, header=None, usecols=[25])
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
