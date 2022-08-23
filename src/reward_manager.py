#!/usr/bin/env python3
# Tim Marvel
import numpy as np
def compute_reward(before, after):
    # both numpy arrays have to be of the same dimensions
    if before.shape != after.shape:
        return -1, False

    shape = before.shape
    total_sum = 0.0
    has_acted = False
    is_only_ten = True
    has_no_ten = True
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            # check for hit a mine state
            if after[i, j] == -1:
                return -0.8, False
            if after[i, j] == 10:
                has_no_ten = False
            if is_only_ten and before[i, j] < 10:
                is_only_ten = False
            if after[i, j] == before[i, j]:
                continue
            if before[i, j] == 10 and (after[i, j] < 10 or after[i, j] == 90):
                total_sum += 1
                has_acted = True
                continue

    if not has_acted:
        total_sum = 0.0
    # print("this is total sum: "+str(total_sum))
    # print("this is before :"+str(before))
    # print("this is after :"+str(after))
    #  if no ten is present this means the board has been cleared and the game has been won :)
    return total_sum, has_no_ten


def reward_shaper_pos_three(reward, grid):
    if reward > 0.5 and grid.sum() == 90:
        reward = 0.25
    elif reward > 0.5 and grid.sum() == 6*10-3:
        reward = 0.25
    elif reward > 0.5 and grid.sum() == (4 * 10 - 5):
        reward = 0.25

    return reward


def reward_shaper_neg_three(reward, grid):
    grid_2d = grid.reshape(3,3)
    if grid_2d[1, 1] == 90:
        reward = -10.0
    elif grid_2d[1, 1] == 0:
        reward = -10.0
    elif reward == -64:
        reward = -64.0
    elif reward == 0:
        reward = -0.15

    return reward


def reward_shaper_pos_five(reward, grid):
    if reward > 0.5 and grid.sum() == 25*10:
        reward = 0.25
    '''
    elif reward > 0.5 and grid.sum() == 6 * 10 - 3:
        reward = 0.25
    elif reward > 0.5 and grid.sum() == (4 * 10 - 5):
        reward = 0.25
    '''
    if reward>5:
        reward = 2
    return reward


def reward_shaper_neg_five(reward, grid):
    if grid[2, 2] == 90:
        reward = -12.0
    elif grid[2, 2] == 0:
        reward = -12.0
    elif reward == -64:
        reward = -64.0
    elif reward == 0:
        reward = -0.15

    return reward


def reward_shaper_three(rewards, grids):
    for i in range(len(rewards)):
        reward=rewards[i]
        grid=grids[i]
        if reward <= 0:
            rewards[i]=reward_shaper_neg_three(reward, grid)
        #else:
            #rewards[i]=reward_shaper_pos_three(reward, grid)

    #print("end")
    return rewards

def reward_shaper_five(rewards, grids):
    for i in range(len(rewards)):
        reward=rewards[i]
        grid=grids[i]
        grid_2d = grid.reshape(5,5)
        if reward <= 0:
            rewards[i]=reward_shaper_neg_five(reward, grid_2d)
        #else:
            #rewards[i]=reward_shaper_pos_three(reward, grid)

    #print("end")
    return rewards