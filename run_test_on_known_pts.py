#!/usr/bin/env python3
# Tim Marvel
import json
import os
import torch
import random
import math
import numpy as np
from numpy import linalg
from time import sleep

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

#random_percent = 0.0
VERBOSE = False
NUM_ACTIONS = 64

three_by_three_data_filename = "test/three_by_three_testdata.txt"

# can read in the test data, specify the neural net size 3 or 5
def readTestDataFile(net_size = 3):
    end_of_data = net_size*net_size
    data_array = []
    rewards = []

    f = open(three_by_three_data_filename, "r")
    for line in f:
        chunks = line.split(',')
        # print(chunks[0:end_of_data])
        # print(chunks[end_of_data])
        data_to_append = np.array(chunks[0:end_of_data]).astype(np.float32)
        data_to_append = data_to_append.reshape(net_size,net_size)
        data_array.append(data_to_append)
        rewards.append(float(chunks[end_of_data]))

    return data_array,rewards

def loadAndPrepareNet(net_name = "saved_nets/raw_net_three_probofchg"):
    main_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayerLarger()
    main_net.load_state_dict(torch.load(net_name, map_location='cpu'))
    main_net.eval()
    return main_net

def run_test(net_size = 3):
    data, rwrds = readTestDataFile()
    test_net = loadAndPrepareNet()
    total_pts = len(data)
    print("testing "+str(total_pts)+" known data points")

    for i in range(total_pts):
        print("--------- at index %d------------" %(i))
        data_t = torch.from_numpy(data[i])
        data_t = data_t.unsqueeze(0)
        data_t = data_t.unsqueeze(0)
        print(data_t[0][0])
        result = test_net.forward(data_t)
        print("Evaluated value: %.2f vs expected value: %.2f "%(result,rwrds[i]))

run_test()