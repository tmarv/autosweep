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
    #print("this is total sum: "+str(total_sum))
    #print("this is before :"+str(before))
    #print("this is after :"+str(after))
    #  if no ten is present this means the board has been cleared and the game has been won :)
    return total_sum, has_no_ten
