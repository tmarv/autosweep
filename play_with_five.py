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

NUM_ACTIONS = 64

# Silence external libs
logging.basicConfig(level=logging.CRITICAL, filename='logs/play_with_five.log', encoding='utf-8')
logger = logging.getLogger('play_with_five')
# enable logs for current lib
logger.setLevel(level=logging.INFO)


def init_mnswpr():
    tools.launch_mines()
    # can be slow
    sleep(1)
    # start 8 by 8 minesweeper
    tools.move_and_click(739, 320)


def select_action_five(neural_net, state, normalize = False, norm_a = 2.0, norm_b = 12.0):
    state = tools.extend_state_five(state)
    score_board = np.zeros((8, 8))
    for i in range(2, 10):
        for j in range(2, 10):
            local_cpy = tools.grab_sub_state_noext_five(state, j, i) # normalize of needed
            if normalize:
                local_cpy = local_cpy.flatten()
                for k in range(25):
                    local_cpy[k] = (local_cpy[k]+norm_a)/norm_b
            local_tensor = torch.from_numpy(local_cpy).to(dtype=torch.float)
            local_tensor = local_tensor.reshape([5, 5])
            local_tensor = local_tensor.unsqueeze(0)
            local_tensor = local_tensor.unsqueeze(0)
            neural_net_pred = neural_net.forward(local_tensor)
            score_board[i - 2, j - 2] = neural_net_pred

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


def play_mnswpr(iterations, net_name, sz = 64 , epoch = '', is_test_set = False, random_percent = 0.0):
    logger.info('playing with net: {}'.format(net_name))
    main_net = model_zoo.FiveByFive1ConvLayerX(sz, 0.0)
    #main_net = model_zoo.FiveByFive2ConvLayerX(sz, 0.0)
    main_net_name = os.path.abspath(
        os.path.join(tools.get_working_dir(), '../saved_nets/{}'.format(net_name)))
    main_net.load_state_dict(torch.load(main_net_name, map_location=device))
    main_net.eval()

    # count how many games are played, won or lost
    i_episode = 0
    win = 0
    lose = 0
    while i_episode < iterations:
        sleep(0.3)
        logger.info("episode {}".format(i_episode))
        tools.move_and_click(739, 320)
        sleep(0.3)
        state = dg.get_state_from_screen()
        # run the flagging algorithm
        state = min_int.mark_game(state)
        # perform a deep copy
        previous_state = state.copy()
        sleep(0.3)
        counter = 0
        while not dg.get_status() and counter < 200:
            action = select_action_five(main_net, state, True)
            counter += 1
            # replace some good shots by randomness to collect more data
            for k in range(0, 64):
                if random.random() < random_percent:
                    action[k][0] = random.randint(0, 7)
                    action[k][1] = random.randint(0, 7)
                min_int.move_and_click_to_ij(action[k][0], action[k][1])
                tools.move_to(1490, 900)
                # if hit a mine
                sleep(0.5)
                state = dg.get_state_from_screen()
                if not dg.get_status():
                    state = min_int.mark_game(state)
                sub_state_three = tools.grab_sub_state_three(previous_state, action[k][1] + 1, action[k][0] + 1)
                sub_state_five = tools.grab_sub_state_five(previous_state, action[k][1] + 2, action[k][0] + 2)
                sub_state_seven = tools.grab_sub_state_seven(previous_state, action[k][1] + 3, action[k][0] + 3)
                #print(sub_state_seven)
                #exit()
                # we hit a mine
                if dg.get_status():
                    logger.info('LOST: hit mine')
                    tools.save_action_neg_three(-64, sub_state_three, is_test_set)
                    tools.save_action_neg_five(-64, sub_state_five, is_test_set)
                    tools.save_action_neg_seven(-64, sub_state_seven, is_test_set)
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
                    tools.save_action_three(10, sub_state_three, is_test_set)
                    tools.save_action_five(10, sub_state_five, is_test_set)
                    tools.save_action_seven(10, sub_state_seven, is_test_set)
                    tools.move_and_click(1490, 900)
                    logger.info('has won in {} strikes'.format(counter))
                    counter += 1001
                    i_episode = i_episode + 1
                    win += 1
                    break
                else:
                    # save data from transition
                    if reward > 0:
                        tools.save_action_three(reward, sub_state_three, is_test_set)
                        tools.save_action_five(reward, sub_state_five, is_test_set)
                        tools.save_action_seven(reward, sub_state_seven, is_test_set)
                    elif reward <= 0:
                        tools.save_action_neg_three(reward, sub_state_three, is_test_set)
                        tools.save_action_neg_five(reward, sub_state_five, is_test_set)
                        tools.save_action_neg_seven(reward, sub_state_seven, is_test_set)

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


device = tools.get_device()
init_mnswpr()
logger.info('-- starting to play --')
logger.info('this is the device: {}'.format(device))

play_mnswpr(iterations=1, sz=32, net_name='five_conv_32_drop_0_bs_64_m25_nd_l2', random_percent = 0.0)
#play_mnswpr(iterations=300, sz=256, net_name='five_conv_256_drop_0_bs_131072_m25_nd_l1', random_percent = 0.0)
#play_mnswpr(iterations=300, sz=512, net_name='five_conv_512_drop_0_bs_131072_m25_nd_l1', random_percent = 0.0)
logger.info('------ finished playing ------')