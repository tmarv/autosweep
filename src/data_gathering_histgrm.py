#!/usr/bin/env python3
# Tim Marvel
import cv2
import numpy as np
import os
import pyscreenshot

HIST_TILE_SIZE = (10, 10)
HIST_BINS = [32, 32, 32]
HIST_LAYERS = [0, 1, 2]
HIST_RANGE = [0, 256, 0, 256, 0, 256]
HIST_COMPARE_MTHD = 0  # opencv method for histogram comparison

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

one_template = cv2.imread(os.path.join(dir_path, "../templates/one_template.png"))
one_template = cv2.resize(one_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_one = cv2.calcHist([one_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_one, hist_one).flatten()

two_template = cv2.imread(os.path.join(dir_path, "../templates/two_template.png"))
two_template = cv2.resize(two_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_two = cv2.calcHist([two_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_two, hist_two).flatten()

three_template = cv2.imread(os.path.join(dir_path, "../templates/three_template.png"))
three_template = cv2.resize(three_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_three = cv2.calcHist([three_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_three, hist_three).flatten()

four_template = cv2.imread(os.path.join(dir_path, "../templates/four_template.png"))
four_template = cv2.resize(four_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_four = cv2.calcHist([four_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_four, hist_four).flatten()

five_template = cv2.imread(os.path.join(dir_path, "../templates/five_template.png"))
five_template = cv2.resize(five_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_five = cv2.calcHist([five_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_five, hist_five).flatten()

six_template = cv2.imread(os.path.join(dir_path, "../templates/six_template.png"))
six_template = cv2.resize(six_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_six = cv2.calcHist([six_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_six, hist_six).flatten()

seven_template = cv2.imread(os.path.join(dir_path, "../templates/seven_template.png"))
seven_template = cv2.resize(seven_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_seven = cv2.calcHist([seven_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_seven, hist_seven).flatten()

eight_template = cv2.imread(os.path.join(dir_path, "../templates/eight_template.png"))
eight_template = cv2.resize(eight_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_eight = cv2.calcHist([eight_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_eight, hist_eight).flatten()

cleared_template = cv2.imread(os.path.join(dir_path, "../templates/cleared_template.png"))
cleared_template = cv2.resize(cleared_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_cleared = cv2.calcHist([cleared_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_cleared, hist_cleared).flatten()

mine_ex_template = cv2.imread(os.path.join(dir_path, "../templates/exploded_mine_template.png"))
mine_ex_template = cv2.resize(mine_ex_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_mine_ex = cv2.calcHist([mine_ex_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_mine_ex, hist_mine_ex).flatten()

mine_template = cv2.imread(os.path.join(dir_path, "../templates/mine_visible_template.png"))
mine_template = cv2.resize(mine_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_mine = cv2.calcHist([mine_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_mine, hist_mine).flatten()

flag_template = cv2.imread(os.path.join(dir_path, "../templates/flag_template.png"))
flag_template = cv2.resize(flag_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_flag = cv2.calcHist([flag_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_flag, hist_flag).flatten()

flag_grey = cv2.cvtColor(flag_template, cv2.COLOR_BGR2GRAY)
hist_flag_grey = cv2.calcHist([flag_grey], [0], None, [8], [0, 80])
cv2.normalize(hist_flag_grey, hist_flag_grey).flatten()

unkw_template = cv2.imread(os.path.join(dir_path, "../templates/unknown_template.png"))
unkw_template = cv2.resize(unkw_template, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
hist_unkw = cv2.calcHist([unkw_template], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
cv2.normalize(hist_unkw, hist_unkw).flatten()

unkw_grey = cv2.cvtColor(unkw_template, cv2.COLOR_BGR2GRAY)
hist_unkw_grey = cv2.calcHist([unkw_grey], [0], None, [8], [0, 80])
cv2.normalize(hist_unkw_grey, hist_unkw_grey).flatten()

best_times = cv2.imread(os.path.join(dir_path, "../templates/best_times.png"))
hist_best_times = cv2.calcHist([best_times], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)

is_exploded = False
is_bestTimes = False
unknown_counter = 0
delta_px = 2
tile_dx = 59


def reset():
    global is_exploded
    is_exploded = False

def gotTopTimes():
    return is_bestTimes

def get_unknown_counter():
    return unknown_counter


def get_status():
    return is_exploded


def get_state_from_screen():
    global is_exploded
    global unknown_counter
    global is_bestTimes
    board_state = np.ones((8, 8))
    is_exploded = False
    unknown_counter = 0
    screen_shot = pyscreenshot.grab(bbox=(0, 0, 1920, 1080))
    numpy_image = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_RGB2BGR)
    smaller_size = (960, 540)
    resized = cv2.resize(numpy_image, smaller_size, interpolation=cv2.INTER_AREA)

    img = resized[47:533, 213:699]

    # check if we have a best times dialog
    best_times_tile = img[102:119, 369:404]
    hist_best_times_tile = cv2.calcHist([best_times_tile], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
    cv2.normalize(hist_best_times_tile, hist_best_times_tile).flatten()
    comp_best_times = cv2.compareHist(hist_best_times, hist_best_times_tile, HIST_COMPARE_MTHD)
    
    if(comp_best_times>0.95):
        is_bestTimes=True
        return

    is_bestTimes = False
    '''
    print("this is best times comp: "+str(comp_best_times))
    cv2.imshow("best_times_tile", best_times_tile)

    cv2.imshow("numpy_image", numpy_image)
    cv2.imwrite("backup40.png", numpy_image)
    
    cv2.imshow("img", img)
    cv2.imwrite("backup.png", img)
    cv2.waitKey()
    exit()
    '''

    for i in range(0, 8):
        for j in range(0, 8):
            x1 = i * (tile_dx + delta_px)
            y1 = j * (tile_dx + delta_px)
            tile = img[y1:y1 + tile_dx, x1:x1 + tile_dx]

            # this error gets triggered if something is wrong with the display
            # it protects against bad data being gathered
            if tile.shape[0] != tile.shape[1]:
                print("error i: " + str(i) + " j: " + str(j) + "   " + str(tile.shape[0]) + "   " + str(tile.shape[1]))
                exit()
            resized = cv2.resize(tile, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
            hist_tile = cv2.calcHist([resized], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
            cv2.normalize(hist_tile, hist_tile).flatten()

            comp_one = cv2.compareHist(hist_one, hist_tile, HIST_COMPARE_MTHD)
            if comp_one > 0.9:
                board_state[j, i] = 1
                continue

            comp_two = cv2.compareHist(hist_two, hist_tile, HIST_COMPARE_MTHD)
            if comp_two > 0.9:
                board_state[j, i] = 2
                continue

            comp_three = cv2.compareHist(hist_three, hist_tile, HIST_COMPARE_MTHD)
            if comp_three > 0.9:
                board_state[j, i] = 3
                continue

            comp_four = cv2.compareHist(hist_four, hist_tile, HIST_COMPARE_MTHD)
            if comp_four > 0.9:
                board_state[j, i] = 4
                continue

            comp_five = cv2.compareHist(hist_five, hist_tile, HIST_COMPARE_MTHD)
            if comp_five > 0.9:
                board_state[j, i] = 5
                continue

            comp_six = cv2.compareHist(hist_six, hist_tile, HIST_COMPARE_MTHD)
            if comp_six > 0.9:
                board_state[j, i] = 6
                continue

            comp_seven = cv2.compareHist(hist_seven, hist_tile, HIST_COMPARE_MTHD)
            if comp_seven > 0.9:
                board_state[j, i] = 7
                continue

            comp_eight = cv2.compareHist(hist_eight, hist_tile, HIST_COMPARE_MTHD)
            if comp_eight > 0.9:
                board_state[j, i] = 8
                continue

            mine_ex_comp = cv2.compareHist(hist_mine_ex, hist_tile, HIST_COMPARE_MTHD)
            mine_comp = cv2.compareHist(hist_mine, hist_tile, HIST_COMPARE_MTHD)
            if mine_ex_comp > 0.9 or mine_comp > 0.9:
                is_exploded = True
                board_state[j, i] = 99
                # TODO look into this
                return board_state

            comp_cleared = cv2.compareHist(hist_cleared, hist_tile, HIST_COMPARE_MTHD)
            if comp_cleared > 0.9:
                board_state[j, i] = 0
                continue

            # this special procedure is needed to distinguish mines from empty cells
            grey_tile = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            grey_tile_hist = cv2.calcHist([grey_tile], [0], None, [8], [0, 80])

            test_hist_comp_flag = cv2.compareHist(grey_tile_hist, hist_flag_grey, 3)

            if test_hist_comp_flag < 0.8:
                board_state[j, i] = 10
                continue
            else:
                board_state[j, i] = -1
                unknown_counter += 1
                continue

    return board_state

def get_state_from_screen_large():
    global is_exploded
    global unknown_counter
    global is_bestTimes
    board_state = np.ones((16, 30))
    is_exploded = False
    unknown_counter = 0
    screen_shot = pyscreenshot.grab(bbox=(0, 0, 1920, 1080))
    numpy_image = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_RGB2BGR)
    smaller_size = (960, 540)
    resized = cv2.resize(numpy_image, smaller_size, interpolation=cv2.INTER_AREA)
    img_res = resized[47:533, 213:699]
    
    height_pts = [0,51,54,106,109,161,164,216,219,271,274,326,329,381,384,436,439,491,494,545,548,599,602,653,656,707,710,761,764,815,818,868]

    width_pts = [0,51,54,106,109,161,164,216,219,271,274,326,329,381,384,436,439,491,494,546,549,601,604,656,659,711,714,766,769,821,824,876,
            879,931,934,986,989,1040,1043,1094,1097,1148,1151,1202,1205,1256,1259,1310,1313,1364,1367,1418,1421,1472,1475,1526,1529,1580,1583,1634]
    
    img = numpy_image[142:1011,92:1726]

    # check if we have a best times dialog
    best_times_tile = img_res[102:119, 369:404]
    hist_best_times_tile = cv2.calcHist([best_times_tile], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
    cv2.normalize(hist_best_times_tile, hist_best_times_tile).flatten()
    comp_best_times = cv2.compareHist(hist_best_times, hist_best_times_tile, HIST_COMPARE_MTHD)
    
    if(comp_best_times>0.95):
        is_bestTimes=True
        return

    is_bestTimes = False

    for i in range (0, len(height_pts), 2):
        for j in range (0, len(width_pts), 2):
            i_floor = int(i/2)
            j_floor = int(j/2)

            left = width_pts[j]
            right = width_pts[j+1]
            top = height_pts[i]
            bottom = height_pts[i+1]
            tile = img[top:bottom, left:right]
            # keep this to check if we hit top score
            resized = cv2.resize(tile, HIST_TILE_SIZE, interpolation=cv2.INTER_CUBIC)
            hist_tile = cv2.calcHist([resized], HIST_LAYERS, None, HIST_BINS, HIST_RANGE)
            cv2.normalize(hist_tile, hist_tile).flatten()

            comp_one = cv2.compareHist(hist_one, hist_tile, HIST_COMPARE_MTHD)
            if comp_one > 0.9:
                board_state[i_floor, j_floor] = 1
                continue

            comp_two = cv2.compareHist(hist_two, hist_tile, HIST_COMPARE_MTHD)
            if comp_two > 0.9:
                board_state[i_floor, j_floor] = 2
                continue

            comp_three = cv2.compareHist(hist_three, hist_tile, HIST_COMPARE_MTHD)
            if comp_three > 0.9:
                board_state[i_floor, j_floor] = 3
                continue

            comp_four = cv2.compareHist(hist_four, hist_tile, HIST_COMPARE_MTHD)
            if comp_four > 0.9:
                board_state[i_floor, j_floor] = 4
                continue

            comp_five = cv2.compareHist(hist_five, hist_tile, HIST_COMPARE_MTHD)
            if comp_five > 0.9:
                board_state[i_floor, j_floor] = 5
                continue

            comp_six = cv2.compareHist(hist_six, hist_tile, HIST_COMPARE_MTHD)
            if comp_six > 0.9:
                board_state[i_floor, j_floor] = 6
                continue

            comp_seven = cv2.compareHist(hist_seven, hist_tile, HIST_COMPARE_MTHD)
            if comp_seven > 0.9:
                board_state[i_floor, j_floor] = 7
                continue

            comp_eight = cv2.compareHist(hist_eight, hist_tile, HIST_COMPARE_MTHD)
            if comp_eight > 0.9:
                board_state[i_floor, j_floor] = 8
                continue

            mine_ex_comp = cv2.compareHist(hist_mine_ex, hist_tile, HIST_COMPARE_MTHD)
            mine_comp = cv2.compareHist(hist_mine, hist_tile, HIST_COMPARE_MTHD)
            if mine_ex_comp > 0.9 or mine_comp > 0.9:
                is_exploded = True
                board_state[i_floor, j_floor] = 99
                # TODO look into this
                return board_state

            comp_cleared = cv2.compareHist(hist_cleared, hist_tile, HIST_COMPARE_MTHD)
            if comp_cleared > 0.9:
                board_state[i_floor, j_floor] = 0
                continue

            # this special procedure is needed to distinguish mines from empty cells
            grey_tile = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            grey_tile_hist = cv2.calcHist([grey_tile], [0], None, [8], [0, 80])

            test_hist_comp_flag = cv2.compareHist(grey_tile_hist, hist_flag_grey, 3)
            if test_hist_comp_flag < 0.8:
                board_state[i_floor, j_floor] = 10
                continue
            else:
                board_state[i_floor, j_floor] = -1
                unknown_counter += 1
                continue

    return board_state
