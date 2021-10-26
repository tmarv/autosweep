#!/usr/bin/env python3
# Tim Marvel
import os

import torch

from src import neural_net_lib, tools


def simple_play(how_many, epoch, steps):
    neural_net = neural_net_lib.ThreeByThreeSig()
    net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_' + str(epoch) + '_' + str(
            steps)))
    neural_net.load_state_dict(torch.load(net_name))

    i_episode = 0
    while i_episode < how_many:
        i_episode += 1
