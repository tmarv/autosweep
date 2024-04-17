#!/usr/bin/env python3
# Tim Marvel
import os
import numpy as np
import pandas as pd
import math
import json
import csv


def open_and_sort_3by3(backup_name_unique)->pd.DataFrame:
    df = pd.read_csv(backup_name_unique)
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


def open_and_sort_5by5(backup_name_unique)->pd.DataFrame:
    df = pd.read_csv(backup_name_unique)
    df.columns.values[0] = 'd1'
    df.columns.values[1] = 'd2'
    df.columns.values[2] = 'd3'
    df.columns.values[3] = 'd4'
    df.columns.values[4] = 'd5'
    df.columns.values[5] = 'd6'
    df.columns.values[6] = 'd7'
    df.columns.values[7] = 'd8'
    df.columns.values[8] = 'd9'
    df.columns.values[9] = 'd10'
    df.columns.values[10] = 'd11'
    df.columns.values[11] = 'd12'
    df.columns.values[12] = 'd13'
    df.columns.values[13] = 'd14'
    df.columns.values[14] = 'd15'
    df.columns.values[15] = 'd16'
    df.columns.values[16] = 'd17'
    df.columns.values[17] = 'd18'
    df.columns.values[18] = 'd19'
    df.columns.values[19] = 'd20'
    df.columns.values[20] = 'd21'
    df.columns.values[21] = 'd22'
    df.columns.values[22] = 'd23'
    df.columns.values[23] = 'd24'
    df.columns.values[24] = 'd25'
    df.columns.values[25] = 'reward'
    df = df.sort_values(by=['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10',
                            'd11','d12','d13','d14','d15','d16','d17','d18','d19',
                            'd20','d21','d22','d23','d24','d25'], ascending=True)
    return df


def perform_comparison_3by3(frame_a, frame_b):
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

def perform_comparison_5by5(frame_a, frame_b):
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
    same_data = same_data and frame_a.d10 == frame_b.d10
    same_data = same_data and frame_a.d11 == frame_b.d11
    same_data = same_data and frame_a.d12 == frame_b.d12
    same_data = same_data and frame_a.d13 == frame_b.d13
    same_data = same_data and frame_a.d14 == frame_b.d14
    same_data = same_data and frame_a.d15 == frame_b.d15
    same_data = same_data and frame_a.d16 == frame_b.d16
    same_data = same_data and frame_a.d17 == frame_b.d17
    same_data = same_data and frame_a.d18 == frame_b.d18
    same_data = same_data and frame_a.d19 == frame_b.d19
    same_data = same_data and frame_a.d20 == frame_b.d20
    same_data = same_data and frame_a.d21 == frame_b.d21
    same_data = same_data and frame_a.d22 == frame_b.d22
    same_data = same_data and frame_a.d23 == frame_b.d23
    same_data = same_data and frame_a.d24 == frame_b.d24
    same_data = same_data and frame_a.d25 == frame_b.d25
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


def remove_duplicates_3by3(backup_name_unique):
    print("started remove_duplicates for 3 by 3")
    data = open_and_sort_3by3(backup_name_unique)
    print("----- data sample -----")
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
        same_result, same_data = perform_comparison_3by3(data.iloc[i], data.iloc[i+1])
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

def remove_duplicates_5by5(backup_name_unique):
    print("started remove_duplicates for 5 by 5")
    data = open_and_sort_5by5(backup_name_unique)
    print("----- data sample -----")
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
        same_result, same_data = perform_comparison_5by5(data.iloc[i], data.iloc[i+1])
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


def rotate_data_3by3(backup_name_unique, backup_name_rotated):
    with open(backup_name_unique) as file_obj:
        csv_obj = csv.reader(file_obj)
        rotated_data = []
        for line in csv_obj:
            reward =  np.float32(line[9])

            grid_values_0 = np.array([line[0:3], line[3:6], line[6:9]])
            grid_values_90 = np.rot90(grid_values_0, 1)
            grid_values_180 = np.rot90(grid_values_0, 2)
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


def rotate_data_5by5(backup_name_unique, backup_name_rotated):
    with open(backup_name_unique) as file_obj:
        csv_obj = csv.reader(file_obj)
        rotated_data = []
        for line in csv_obj:
            reward =  np.float32(line[25])

            grid_values_0 = np.array([line[0:5], line[5:10], line[10:15], line[15:20], line[20:25]])
            grid_values_90 = np.rot90(grid_values_0, 1)
            grid_values_180 = np.rot90(grid_values_0, 2)
            grid_values_270 = np.rot90(grid_values_0, 3)

            grid_values_0 = grid_values_0.flatten()
            grid_values_0 = np.append(grid_values_0, line[25])

            grid_values_90 = grid_values_90.flatten()
            grid_values_90 = np.append(grid_values_90, line[25])

            grid_values_180 = grid_values_180.flatten()
            grid_values_180 = np.append(grid_values_180, line[25])

            grid_values_270 = grid_values_270.flatten()
            grid_values_270 = np.append(grid_values_270, line[25])

            rotated_data.append(grid_values_0)
            rotated_data.append(grid_values_90)
            rotated_data.append(grid_values_180)
            rotated_data.append(grid_values_270)

        dump_to_file(rotated_data, backup_name_rotated)


def normalize_rewards_and_inputs(read_name, output_name, index, min_val_i = -100, max_val_i = 100):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_rewards = []
        raw_data = []
        raw_lines = []

        for line in csv_obj:
            #adjusting the reward for a smoother spread
            adjusted_val = max(min_val_i, np.float32(line[index]))
            adjusted_val = min(max_val_i, adjusted_val)
            raw_rewards.append(adjusted_val)
            line[index] = adjusted_val
            raw_lines.append(np.array(line))
            for i in range(index):
                raw_data.append(np.array(np.float32(line[i])))

        min_val_r = np.min(raw_rewards)
        max_val_r = np.max(raw_rewards)
        print("this is min val reward: "+str(min_val_r))
        print("this is max val reward: "+str(max_val_r))

        min_val_d = np.min(raw_data)
        max_val_d = np.max(raw_data)
        print("this is min val data: " + str(min_val_d))
        print("this is max val data: " + str(max_val_d))

        for line in raw_lines:
            scaled_reward = (np.float32(line[index]) - min_val_r) / (max_val_r - min_val_r)
            line[index] = scaled_reward
            for i in range(index):
                line[i] = (np.float32(line[i]) - min_val_d) / (max_val_d - min_val_d)

        print('there are {} normalized lines'.format(len(raw_lines)))
        dump_to_file(raw_lines, output_name)


def run_pre_processing_three():
    print("starting data pre-processing of 3 by 3 data")
    rotate_data_3by3("rewards3.txt", "rotated_pts3.csv")
    remove_duplicates_3by3("rotated_pts3.csv")
    normalize_rewards_and_inputs("rotated_pts3.csv", "unique_normalized_3_rewards_m25.csv", 9, -2, 5)


def run_pre_processing_five():
    print("starting data pre-processing of 5 by 5 data")
    rotate_data_5by5("rewards5.txt", "rotated_pts5.csv")
    remove_duplicates_5by5("rotated_pts5.csv")
    normalize_rewards_and_inputs("rotated_pts5.csv", "unique_normalized_5_rewards_m25.csv", 25, -2, 5)

run_pre_processing_five()