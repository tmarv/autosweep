#!/usr/bin/env python3
# Tim Marvel
import os

import torch

from src import neural_net_lib, tools

from time import sleep


def play_with_three(how_many, epoch, steps):
    neural_net = neural_net_lib.ThreeByThreeSig()
    net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_' + str(epoch) + '_' + str(
            steps)))
    neural_net.load_state_dict(torch.load(net_name))

    i_episode = 0
    while i_episode < how_many:
        i_episode += 1


def play_with_five(how_many, epoch):
    print("starting only five by five play")
    neural_net = neural_net_lib.FiveByFiveSig()
    net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/neural_net_five_' + str(epoch)))
    neural_net.load_state_dict(torch.load(net_name))
    i_episode = 0
    while i_episode < how_many:
        sleep(0.3)
        print("this is i_episode " + str(i_episode))
        tools.move_and_click(739, 320)
        sleep(0.3)
        state = dg.get_state_from_screen()
        # run the flagging algorithm
        state = min_int.mark_game(state)
        sleep(0.3)
        counter = 0