#!/usr/bin/env python3
# Tim Marvel
import json
import logging
import math
import numpy as np
from numpy import linalg
import os
import random
import torch
from time import sleep

from src import data_gathering_histgrm as dg
from src import reward_manager
from src import minesweeper_interface as min_int
from src import model_zoo
from src import tools


# Silence external libs
logging.basicConfig(level=logging.CRITICAL, filename='logs/play_with_seven.log')
logger = logging.getLogger('play_with_seven')
# enable logs for current lib
logger.setLevel(level=logging.INFO)


def init_mnswpr():
    tools.launch_mines()
    # can be slow
    sleep(1)
    # start 8 by 8 minesweeper
    tools.move_and_click(739, 320)



def skip_empty(local_cpy):
    inner_3by3 = local_cpy[[2, 3, 4], :][:, [2, 3, 4]]
    if -1 in inner_3by3:
        return False
    return True


def select_action_seven(neural_net, state, normalize = False, norm_a = 2.0, norm_b = 12.0):
    state = tools.extend_state_seven(state)
    score_board = np.zeros((8, 8))
    for i in range(3, 11):
        for j in range(3, 11):
            local_cpy = tools.grab_sub_state_noext_seven(state, j, i)
            # avoid going to empty places/ running inference on empty
            if skip_empty(local_cpy):
                continue
            if normalize:
                local_cpy = local_cpy.flatten()
                for k in range(49):
                    local_cpy[k] = (local_cpy[k]+norm_a)/norm_b
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.reshape([7, 7])
            local_tensor = local_tensor.unsqueeze(0)
            local_tensor = local_tensor.unsqueeze(0)
            neural_net_pred = neural_net.forward(local_tensor)
            score_board[i - 3, j - 3] = neural_net_pred

    flat = score_board.flatten()
    flat.sort()
    flat = np.flipud(flat)
    return_values = []
    total_len = len(flat)
    i = 0
    while i < total_len:
        local_range = np.where(score_board == flat[i])
        local_sz = len(local_range[0])
        for j in range(0, local_sz):
            return_values.append([local_range[0][j], local_range[1][j]])
            i = i + 1
    if len(return_values) != 64:
        print("Catastrophic error: return values size is off " + str(len(return_values)))
        logger.critical('Catastrophic error: return values size is off: {}'.format(len(return_values)))
        exit()
    return return_values


def play_mnswpr(iterations, net_name, sz = 64 ):
    logger.info('playing with net: {}'.format(net_name))
    # main_net = model_zoo.SevenBySeven1ConvLayerXLeakyReLU(sz, 0.0)
    # main_net = model_zoo.SevenBySeven1ConvLayerXLeakyReLUSigmoidEnd(sz, 0.0)
    main_net = model_zoo.SevenBySeven2ConvLayerXLeakyReLUSigmoidEnd(sz, 0.0)
    main_net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/{}'.format(net_name)))
    main_net.load_state_dict(torch.load(main_net_name, map_location=device))
    main_net.eval()

    # count how many games are played, won or lost
    i_episode = 0
    win = 0
    lose = 0
    while i_episode < iterations:
        sleep(0.1)
        logger.info("episode {}".format(i_episode))
        tools.move_and_click(739, 320)
        sleep(0.1)
        state = dg.get_state_from_screen()
        if dg.gotTopTimes():
            tools.move_and_click(1200,320)
            sleep(0.1)
            state = dg.get_state_from_screen()
        # run the flagging algorithm
        state = min_int.mark_game(state)
        # perform a deep copy
        previous_state = state.copy()
        sleep(0.05)
        counter = 0
        while not dg.get_status() and counter < 500:
            action = select_action_seven(main_net, state, True)
            counter += 1
            # replace some good shots by randomness to collect more data
            for k in range(0, 64):
                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                tools.move_to(1490, 900)
                # if hit a mine or got into best times
                sleep(0.05)
                state = dg.get_state_from_screen()
                if dg.gotTopTimes():
                    tools.move_and_click(1200, 320)
                    sleep(0.05)
                    state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[k][1] + 2, action[k][0] + 2)
                sub_state_seven = tools.grab_sub_state_seven(previous_state, action[k][1] + 3, action[k][0] + 3)
                # we hit a mine
                if dg.get_status():
                    logger.info('LOST: hit mine')
                    logger.info(' \n '+str(sub_state_three))
                    logger.info(' \n '+str(sub_state_five))
                    tools.move_and_click(1490, 900)
                    counter += 1
                    i_episode = i_episode + 1
                    lose += 1
                    break

                # compute reward
                reward, has_won = reward_manager.compute_reward(previous_state, state)
                if has_won:
                    tools.move_and_click(1490, 900)
                    logger.info('has won in {} strikes'.format(counter))
                    counter += 1001
                    i_episode = i_episode + 1
                    win += 1
                    break

                previous_state = state.copy()
                state = min_int.mark_game(state)
                # if we didn't act -> k = 100
                if reward != 0:
                    break
                counter += 1
    
    logger.info('won {} games'.format(win))
    logger.info('lost {} games'.format(lose))
    print('won {} games'.format(win))
    print('lost {} games'.format(lose))


# cpu inference is faster for small batches
device = 'cpu'
init_mnswpr()
logger.info('-- starting to play --')
logger.info('this is the device: {}'.format(device))
print('this is the device: {}'.format(device))


play_mnswpr(iterations=1, sz=16, net_name='seven_conv_16_drop_0_bs_64_m25_nd_l1')
#play_mnswpr(iterations=300, sz=16, net_name='seven_conv_16_drop_0_bs_64_m25_nd_l1')
#play_mnswpr(iterations=500, sz=32, net_name='seven_conv_32_drop_0_bs_128_m25_nd_l1')
#play_mnswpr(iterations=200, sz=32, net_name='seven_conv_32_drop_0_bs_128_m25_nd_l1')
#play_mnswpr(iterations=100, sz=32, net_name='seven_conv_32_drop_0_bs_128_m25_nd_l1')

logger.info('------ finished playing ------')