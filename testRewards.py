#!/usr/bin/env python3
# Tim Marvel
import os
import torch
import random
import numpy as np
from time import sleep

# self made classes
from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import neural_net_lib
from src import tools

def perform_eval(test_case,neural_net_three):
    test_case = torch.from_numpy(test_case).to(dtype=torch.float)
    test_case = test_case.unsqueeze(0)
    result_corner = neural_net_three.forward(test_case.reshape([1, 9]))
    return  result_corner

def play_with_three(how_many, epoch, steps):
    # load the 3 by 3 kernel network
    neural_net_three = neural_net_lib.ThreeByThreeSig()
    # TODO: fix this
    net_name_three = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_three_test_' + str(epoch)))
    device = 'cpu'
    neural_net_three.load_state_dict(torch.load(net_name_three, map_location=device))
    neural_net_three.eval()

    test_corner = np.array([[-1.,10.,10.],[-1.,10.,10.],[-1.,-1.,-1.]])
    test_corner2 = np.array([[-1.,-1.,-1.],[-1.,10.,10.],[-1.,10.,10.]])
    test_side = np.array([[-1.,-1.,-1.],[10.,10.,10.],[10.,10.,10.]])
    test_empty = np.array([[10,10,10],[10,10,10],[10,10,10]])

    print(perform_eval(test_corner,neural_net_three))
    print(perform_eval(test_corner2,neural_net_three))
    print("sides:")
    print(perform_eval(test_side,neural_net_three))
    print(perform_eval(test_empty,neural_net_three))


play_with_three(3,3,3)