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
    net_name_three = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_three_test_clean_short'))
    device = tools.get_device()
    neural_net_three.load_state_dict(torch.load(net_name_three, map_location=device))
    neural_net_three.eval()

    test_corner = np.array([[-1.,10.,10.],[-1.,10.,10.],[-1.,-1.,-1.]])
    test_corner2 = np.array([[-1.,-1.,-1.],[-1.,10.,10.],[-1.,10.,10.]])
    test_corner3 = np.array([[10.,10.,-1],[10, 10.,-1.],[-1.,-1.,-1.]])
    test_corner4 = np.array([[-1.,-1.,-1.],[10.,10.,-1],[10, 10.,-1.]])

    test_side = np.array([[-1.,-1.,-1.],[10.,10.,10.],[10.,10.,10.]])
    test_empty = np.array([[10,10,10],[10,10,10],[10,10,10]])

    print(perform_eval(test_corner,neural_net_three))
    print(perform_eval(test_corner2,neural_net_three))
    print(perform_eval(test_corner3,neural_net_three))
    print(perform_eval(test_corner4,neural_net_three))
    print("sides:")
    print(perform_eval(test_side,neural_net_three))
    print(perform_eval(test_empty,neural_net_three))


def init_the_cluster_nets(base_name):
    net_0 = neural_net_lib.ThreeByThreeSig()
    net_1 = neural_net_lib.ThreeByThreeSig()
    net_2 = neural_net_lib.ThreeByThreeSig()

    net_0_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/'+base_name+"0"))
    net_1_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/'+base_name+"1"))
    net_2_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/'+base_name+"2"))

    net_0.load_state_dict(torch.load(net_0_name, map_location=device))
    net_1.load_state_dict(torch.load(net_1_name, map_location=device))
    net_2.load_state_dict(torch.load(net_2_name, map_location=device))
    net_0.eval()
    net_1.eval()
    net_2.eval()
    return [net_0,net_1,net_2]

def play_with_cluster():

    #nets_clusters = init_the_cluster_nets("net_three_cluster_")
    # load the 3 by 3 kernel network
    cluster_net = neural_net_lib.ThreeByThreeCluster()
    net_name_cluster = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/cluster_net_three'))
    device = tools.get_device()
    cluster_net.load_state_dict(torch.load(net_name_cluster, map_location=device))
    cluster_net.eval()

    test_corner = np.array([[-1.,10.,10.],[-1.,10.,10.],[-1.,-1.,-1.]])
    test_corner2 = np.array([[-1.,-1.,-1.],[-1.,10.,10.],[-1.,10.,10.]])
    test_corner3 = np.array([[10.,10.,-1],[10, 10.,-1.],[-1.,-1.,-1.]])
    test_corner4 = np.array([[-1.,-1.,-1.],[10.,10.,-1],[10, 10.,-1.]])

    test_side = np.array([[-1.,-1.,-1.],[10.,10.,10.],[10.,10.,10.]])
    test_empty = np.array([[10,10,10],[10,10,10],[10,10,10]])

    test_rewarding = np.array([[10, 10, 10], [1, 0, 0], [0, 0, 0]])

    #print(perform_eval(test_corner,neural_net_three))
    print(perform_eval(test_corner,cluster_net))
    print(torch.argmax(perform_eval(test_corner,cluster_net)).item())
    #print(perform_eval(test_corner2,neural_net_three))
    print(perform_eval(test_corner2,cluster_net))
    #print(perform_eval(test_corner3,neural_net_three))
    print(perform_eval(test_corner3,cluster_net))
    #print(perform_eval(test_corner4,neural_net_three))
    print(perform_eval(test_corner4,cluster_net))
    print("sides:")
    #print(perform_eval(test_side,neural_net_three))
    print(perform_eval(test_side,cluster_net))
    #print(perform_eval(test_empty,neural_net_three))
    print(perform_eval(test_empty,cluster_net))
    print("rewarding cases: ")
    print(perform_eval(test_rewarding,cluster_net))


#play_with_three(3,3,3)
play_with_cluster()