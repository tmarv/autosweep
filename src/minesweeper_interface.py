#!/usr/bin/env python3
# Tim Marvel

import pyautogui
from time import sleep

start_x = 485
start_y = 153
width_extra = 123
height_extra = 123

# TODO check this on the proper distro
# warning this is not maintained, use 20.04 is best
'''
#ubuntu 18.04
start_x = 478
start_y = 147
width_extra = 123
height_extra = 123

#ubuntu 16.04
start_x = 461
start_y = 84
width_extra = 128
height_extra = 128
'''

width = 8
height = 8


def move_and_click_to_ij(i, j):
    pyautogui.moveTo(start_x + i * width_extra, start_y + j * height_extra)
    pyautogui.click()
    pyautogui.moveTo(10, 10)
    sleep(0.03)


def move_and_click_right_to_ij(i, j):
    pyautogui.moveTo(start_x + i * width_extra, start_y + j * height_extra)
    pyautogui.click(button='right')
    pyautogui.moveTo(10, 10)
    sleep(0.03)


def get_min_max_ranges(i, j):
    w_min = -1
    w_max = 2
    h_min = -1
    h_max = 2

    if i - 1 < 0:
        w_min = 0

    if j - 1 < 0:
        h_min = 0

    if i + 1 >= width:
        w_max = 1

    if j + 1 >= height:
        h_max = 1

    return [w_min, w_max, h_min, h_max]


def click_on_cells(i, j, state, num):
    marked_cells = 0
    unmarked_cells = 0
    ranges = get_min_max_ranges(i, j)
    for k in range(ranges[0], ranges[1]):
        for l in range(ranges[2], ranges[3]):
            if k == 0 and l == 0:
                continue

            if state[i + k, j + l] == 10:
                unmarked_cells += 1

            if state[i + k, j + l] == 20:
                marked_cells += 1

            if marked_cells > num:
                return False

    if marked_cells == num and unmarked_cells > 0:
        move_and_click_to_ij(j, i)
        return True

    return False


def mark_cells(i, j, state, num):
    marked_cells = 0
    last_ind = []
    ranges = get_min_max_ranges(i, j)
    for k in range(ranges[0], ranges[1]):
        for l in range(ranges[2], ranges[3]):
            if k == 0 and l == 0:
                continue

            if state[i + k, j + l] == 10:
                marked_cells += 1
                last_ind.append([i + k, j + l])

            if state[i + k, j + l] == 20:
                marked_cells += 1

            if marked_cells > num:
                return False, []

    if marked_cells == num and len(last_ind) > 0:
        for index in last_ind:
            move_and_click_right_to_ij(index[1], index[0])
        return True, last_ind

    return False, []


def mark_game(field_state):
    for j in range(0, height):
        for i in range(0, width):
            current_state = field_state[j, i]
            if 0 < current_state < 9:
                is_flag, last_ind = mark_cells(j, i, field_state, current_state)
                if is_flag:
                    for index in last_ind:
                        field_state[index[0], index[1]] = 20
    return field_state
