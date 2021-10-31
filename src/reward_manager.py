#!/usr/bin/env python3
# Tim Marvel

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
            if before[i, j] == 10 and after[i, j] < 10:
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
    if grid[1, 1] == 90:
        reward = -10.0
    elif grid[1, 1] == 0:
        reward = -10.0
    elif reward == -10:
        reward = -200.0
    elif reward == 0:
        reward = -0.15

    return reward


def reward_shaper_pos_five(sub_state, grid):
    return 1.0


def reward_shaper_neg_five(sub_state, grid):
    return -1.0
