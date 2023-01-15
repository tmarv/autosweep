#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import pandas as pd
import csv
# import torch
# from torch.utils.data import Dataset
# import src.tools
# from src import reward_manager, tools
#import reward_manager, tools


def open_and_read(filename5):
    print(filename5)
    filename_new = "../collected_data/rewards5_only3.txt"
    csv_file = open(filename_new, 'w')
    csvwriter = csv.writer(csv_file)
    dataPoints = pd.read_csv(filename5, header=None,
                                  usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    dataPoints = np.array(dataPoints).astype(np.float32)
    print(dataPoints[0])
    dataPtsRes = []
    for pt in dataPoints:
        if pt[25] == 3:
            dataPtsRes.append(pt)
            csvwriter.writerow(pt)

    print(len(dataPtsRes))
    csv_file.close()
    print("opened file")
'''
class CustomDatasetFromTextFiles5(Dataset):

    def __init__(self, is_small=False, is_clean=False, with_var=False, cluster_num=-2, device="cpu"):
        self.filename5 = tools.get_text_file_names()[1]
        self.augment_data = True
        if is_small:
            self.filename5 = tools.get_text_file_names_small()[1]
        if is_clean:
            self.filename5 = tools.get_text_file_names_clean()[1]
        if with_var:
            self.filename5 = tools.get_text_file_names_var()[1]
        print(self.filename5)

        self.dataPoints = pd.read_csv(self.filename5, header=None,
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        self.dataPoints = np.array(self.dataPoints).astype(np.float32)
        self.rewards = pd.read_csv(self.filename5, header=None, usecols=[25])
        self.rewards = np.array(self.rewards).astype(np.float32)
        self.dataset_size = len(self.dataPoints)
        self.with_cluster = False
        # if we are training for a specific cluster
        if cluster_num > -2:
            self.clusters = np.array(pd.read_csv(self.filename5, header=None, usecols=[27]))
            self.clusters = np.array(self.clusters).astype(np.float32)
            self.rewards = np.array(pd.read_csv(self.filename5, header=None, usecols=[27]))
            self.rewards = np.array(self.rewards).astype(np.float32)
            self.with_cluster = True
            if cluster_num > -1:
                self.rewards = pd.read_csv(self.filename5, header=None, usecols=[25])
                self.rewards = np.array(self.rewards).astype(np.float32)
                self.with_cluster = False
                clustered_data = []
                cluster_rewards = []
                for i in range(self.dataset_size):
                    if self.clusters[i] == cluster_num:
                        clustered_data.append(self.dataPoints[i])
                        cluster_rewards.append(self.rewards[i])
                self.dataPoints = np.array(clustered_data).astype(np.float32)
                self.rewards = np.array(cluster_rewards)
                self.dataset_size = len(self.dataPoints)
        # perform reward shaping here if
        if not self.with_cluster:
            #TODO create reward shaper five
            self.rewards = reward_manager.reward_shaper_five(self.rewards, self.dataPoints)
        if not with_var:
            self.rewards, self.dataPoints = tools.augment_data_five(self.rewards, self.dataPoints)
            self.dataset_size = len(self.rewards)
            print(self.dataset_size)
        if self.with_cluster:
            enlarged_cluster = []
            # print("len before: "+str(len(clusters)))
            for i in range(len(self.rewards)):
                if self.rewards[i] == 0:
                    enlarged_cluster.append(np.array([1, 0, 0]))
                elif self.rewards[i] == 1:
                    enlarged_cluster.append(np.array([0, 1, 0]))
                elif self.rewards[i] == 2:
                    enlarged_cluster.append(np.array([0, 0, 1]))
                else:
                    print("ERROR")
                    print("unkown cluster: " + str(self.clusters[i]))
            self.rewards = np.asarray(enlarged_cluster).astype(np.float32)
        for dataPt in self.dataPoints:
            #TODO: fix/implement this
            # print(dataPt)
            dataPt=dataPt.reshape(5, 5)
        self.rewards = torch.from_numpy(self.rewards).to(device)
        self.dataPoints = torch.from_numpy(self.dataPoints).to(device)
        print("this is dataset size: "+str(self.dataset_size))
    def __getitem__(self, index):
        return self.dataPoints[index], self.rewards[index]

    def __len__(self):
        return self.dataset_size
'''

#dataSet = CustomDatasetFromTextFiles5()
#print(dataSet.__len__())
#print("----")
#print(dataSet.__getitem__(0))
filename5 = "../collected_data/rewards5.txt"
open_and_read(filename5)
print("finished")