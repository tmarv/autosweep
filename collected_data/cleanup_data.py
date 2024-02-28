#!/usr/bin/env python3
# Tim Marvel
import os
import numpy as np
import pandas as pd
import math
import json
import csv


def open_and_sort()->pd.DataFrame:
    df = pd.read_csv("rewards3.txt")
    df.columns.values[0] = 'd1'
    df.columns.values[1] = 'd2'
    df.columns.values[2] = 'd3'
    df.columns.values[3] = 'd4'
    df.columns.values[4] = 'd5'
    df.columns.values[5] = 'd6'
    df.columns.values[6] = 'd7'
    df.columns.values[7] = 'd8'
    df.columns.values[8] = 'd9'
    df.columns.values[9] = 'reward'
    df = df.sort_values(by=['d1','d2','d3','d4','d5','d6','d7','d8','d9'], ascending=True)
    return df


def perform_comparison(frame_a, frame_b):
    same_result = frame_a.reward == frame_b.reward
    same_data = frame_a.d1 == frame_b.d1
    same_data = same_data and frame_a.d2 == frame_b.d2
    same_data = same_data and frame_a.d3 == frame_b.d3
    same_data = same_data and frame_a.d4 == frame_b.d4
    same_data = same_data and frame_a.d5 == frame_b.d5
    same_data = same_data and frame_a.d6 == frame_b.d6
    same_data = same_data and frame_a.d7 == frame_b.d7
    same_data = same_data and frame_a.d8 == frame_b.d8
    same_data = same_data and frame_a.d9 == frame_b.d9
    return same_result, same_data


def dump_to_file(dataList, filename):
    csv_file = open(filename,'w')
    for pdframe in dataList:
        df = pdframe.transpose()
        data_line = ''
        for val in df:
            data_line+=str(val)+","
        csv_file.write(data_line[:-1])
        csv_file.write("\n")
    csv_file.close()


def remove_duplicates(backup_name_unique):
    print("started remove_duplicates for 3 by 3")
    data = open_and_sort()
    print(data)
    if data.empty:
        print("WARNING: the data frame read from the text file is empty")
        return
    total_len = data.shape[0]

    unique_datapts = []
    current_row = data.iloc[0]

    multiple_results = []
    known_rewards = []
    known_rewards.append(current_row.reward)
    unique_datapts.append(current_row)
    cntr = 1
    all_same_cntr = 1
    total_reward = 0
    reward_cntr = 1
    # counts the largest amount of different results for same data in the dataset
    max_cntr = 1
    previous_max = []
    # holds the data for the most common
    most_common_data_diff_result = 0
    for i in range (0, total_len-1):
        same_result, same_data = perform_comparison(data.iloc[i], data.iloc[i+1])
        if same_result and same_data:
            total_reward += data.iloc[i].reward
            reward_cntr += 1
            continue
        elif same_data:
            reward_lcl = data.iloc[i].reward
            total_reward += reward_lcl
            reward_cntr += 1
            if reward_lcl in known_rewards:
                continue
            multiple_results.append(data.iloc[i])
            known_rewards.append(reward_lcl)
        else:
            if max_cntr < reward_cntr:
                most_common_data_diff_result = current_row
                max_cntr = reward_cntr
                previous_max.append(max_cntr)
            if reward_cntr > 1:
                avg_reward = total_reward/reward_cntr
                data.iloc[i].reward = avg_reward
                unique_datapts.append(data.iloc[i])
                known_rewards = []
                known_rewards.append(data.iloc[i+1].reward)
                total_reward = 0
                reward_cntr = 1
            elif reward_cntr == 1:
                unique_datapts.append(data.iloc[i])
                known_rewards = []
                known_rewards.append(data.iloc[i+1].reward)
                total_reward = 0
                reward_cntr = 1
    dump_to_file(unique_datapts, backup_name_unique)
    dump_to_file(multiple_results, "repeats_pts.csv")


def rotate_data(backup_name_unique, backup_name_rotated):
    with open(backup_name_unique) as file_obj:
        csv_obj = csv.reader(file_obj)
        rotated_data = []
        for line in csv_obj:
            reward =  np.float32(line[9])

            grid_values_0 = np.array([line[0:3], line[3:6], line[6:9]])
            grid_values_90 = np.rot90(grid_values_0,1)
            grid_values_180 = np.rot90(grid_values_0,2)
            grid_values_270 = np.rot90(grid_values_0, 3)

            grid_values_0 = grid_values_0.flatten()
            grid_values_0 = np.append(grid_values_0, line[9])

            grid_values_90 = grid_values_90.flatten()
            grid_values_90 = np.append(grid_values_90, line[9])

            grid_values_180 = grid_values_180.flatten()
            grid_values_180 = np.append(grid_values_180, line[9])

            grid_values_270 = grid_values_270.flatten()
            grid_values_270 = np.append(grid_values_270, line[9])

            rotated_data.append(grid_values_0)
            rotated_data.append(grid_values_90)
            rotated_data.append(grid_values_180)
            rotated_data.append(grid_values_270)

        dump_to_file(rotated_data, backup_name_rotated)
print("starting data comparison")
#remove_duplicates("unique_pts.csv")
rotate_data("unique_pts.csv","unique_rotated_pts.csv")
