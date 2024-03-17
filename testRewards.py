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
    test_case = test_case.unsqueeze(0)
    #result_corner = neural_net_three.forward(test_case.reshape([1, 9]))
    result_corner = neural_net_three.forward(test_case)
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


def select_action_cluster(cluster_net, the_nets, state):
    cluster = perform_eval(state, cluster_net)[0]
    #print(clusters)
    #state_three = tools.extend_state(state)
    #score_board = np.zeros((8, 8))

    #for i in range(1, 9):
    #for j in range(1, 9):
    #print("cluster alone")
    #print(cluster)
    mult0 = 1.0
    mult1 = 1.0
    mult2 = 1.0
    if cluster[0]<0.1:
        mult0 = 0
    if cluster[1]<0.1:
        mult1 = 0
    if cluster[2]<0.1:
        mult2 = 0

    result0 = perform_eval(state,the_nets[0])
    result1 = perform_eval(state,the_nets[1])
    result2 = perform_eval(state,the_nets[2])

    score_board = result0*mult0+(result1*mult1)-abs(result2*mult2)
    return score_board

def play_with_cluster():
    nets_clusters = init_the_cluster_nets("net_three_cluster_")
    # load the 3 by 3 kernel network
    #cluster_net = neural_net_lib.ThreeByThreeProbofchng1ConvLayer()
    cluster_net = neural_net_lib.ThreeByThree1ConvLayer512(0.0)
    net_name_cluster = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/three_conv_512_no_drop_bs_2048'))
    device = tools.get_device()
    cluster_net.load_state_dict(torch.load(net_name_cluster, map_location=device))
    cluster_net.eval()

    test_corner = np.array([[-1.,10.,10.],[-1.,10.,10.],[-1.,-1.,-1.]])
    test_corner2 = np.array([[-1.,-1.,-1.],[-1.,10.,10.],[-1.,10.,10.]])
    test_corner3 = np.array([[10.,10.,-1],[10, 10.,-1.],[-1.,-1.,-1.]])
    test_corner3_ = np.array([[1.,10.,10],[10, 10.,10.],[-1.,-1.,-1.]])
    test_corner4 = np.array([[-1.,-1.,-1.],[10.,10.,-1],[10, 10.,-1.]])

    test_side = np.array([[-1.,-1.,-1.],[10.,10.,10.],[10.,10.,10.]])
    test_side2 = np.array([[10.,10.,-1.],[10.,10.,-1.],[10.,10.,-1.]])
    test_empty = np.array([[10,10,10],[10,10,10],[10,10,10]])
    test_empty = np.array([[1,10,10],[2,10,10],[1,10,10]])
    #test_cluster = np.array([[-1,0,0],[-1,0,0],[0,0,0]])
    test_cluster = np.array([[-1.0,0.0,0.0],[-1.0,0.0,0.0],[-1.0,0.0,0.0]])
    '''
        1.0,10.0,10.0,2.0,10.0,10.0,1.0,10.0,10.0,0.6973379
        10.0,10.0,10.0,10.0,10.0,10.0,1.0,2.0,1.0,0.6973379
        10.0,10.0,1.0,10.0,10.0,2.0,10.0,10.0,1.0,0.6973379
        1.0,2.0,1.0,10.0,10.0,10.0,10.0,10.0,10.0,0.6973379
    '''
    test_two1two_side = np.array([[1.0,10.0,10.0],[2.0,10.0,10.0],[1.0,10.0,10.0]])
    test_two1two = np.array([[ 1. ,2., 1.],[ 10., 10., 10.],[10., 10., 10.]])
    #[ 0.9632585   0.07715082 -0.0341807 ]
    test_cluster = np.array([[ 1., 0., -1.],[2.,1.,-1.],[10.,10.,-1.]])

    #test_empty = np.array([[4,10,10],[10,10,10],[10,10,10]])
    #test_empty = np.array([[1,10,-1],[2,10,-1],[10,10,-1]])


    test_rewarding = np.array([[10, 10, 10], [1, 0, 0], [0, 0, 0]])
    test_rewarding2 = np.array([[10, 10, 1], [10, 10, 2], [10, 10, 1]])
    test_rewardig_false = np.array([[1, 10, 10], [10, 10, 10], [-1, -1, -1]])
    test_rewardig_false2 = np.array([[10., 10., -1.], [10., 10., -1.], [10., 10., -1.]])
    '''
    [ 0.37024987 -0.29518628  0.92714]
    [[ 2.  2.  3.],[10. 10. 10.],[-1. -1. -1.]]
    0.37024986743927 -0.29518628120422363 0.9271399974822998
    tensor([-141.2553], grad_fn=<SubBackward0>)
    '''
    test_cluster = np.array([[2.,2.,3.], [10., 10., 10.], [-1., -1., -1.]])
    test_cluster = np.array([[1., 2., 1.], [10., 10., 10.], [10., 10., 10.]])
    test_cluster = np.array([[10.0, 10.0, 1.0], [10.0, 10.0, 2.0], [10.0, 10.0, 1.0]])
    test_ninety = np.array([[1.,1.,1.], [1., 20., 2.], [2., 2., 20.]])
    test_ninety2 = np.array([[1.,1.,1.], [1., 20., 2.], [2., 3., 20.]])

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
    print(perform_eval(test_side2,cluster_net))
    print("test two one two")
    print(perform_eval(test_two1two,cluster_net))
    print(perform_eval(test_two1two_side,cluster_net))
    print("----")
    print("test ninety")
    print(perform_eval(test_ninety,cluster_net))
    print(perform_eval(test_ninety2,cluster_net))

device = tools.get_device()

#play_with_three(3,3,3)
play_with_cluster()
