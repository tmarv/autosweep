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

def open_and_sort(backup_name_unique)->pd.DataFrame:
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
    data = open_and_sort(backup_name_unique)
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


def normalize_rewards_and_inputs(read_name, output_name, min_val_i = -100, max_val_i = 100):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_rewards = []
        raw_data = []
        raw_lines = []

        for line in csv_obj:
            #adjusting the reward for a smoother spread
            adjusted_val = max(min_val_i, np.float32(line[9]))
            adjusted_val = min(max_val_i, adjusted_val)
            raw_rewards.append(adjusted_val)
            line[9] = adjusted_val
            raw_lines.append(np.array(line))
            for i in range(9):
                raw_data.append(np.array(np.float32(line[i])))

        min_val_r = np.min(raw_rewards)
        max_val_r = np.max(raw_rewards)
        print("this is min val r: "+str(min_val_r))
        print("this is max val r: "+str(max_val_r))

        min_val_d = np.min(raw_data)
        max_val_d = np.max(raw_data)
        print("this is min val d: " + str(min_val_d))
        print("this is max val d: " + str(max_val_d))

        for line in raw_lines:
            scaled_reward = (np.float32(line[9]) - min_val_r) / (max_val_r - min_val_r)
            line[9] = scaled_reward
            for i in range(9):
                line[i] = (np.float32(line[i]) - min_val_d) / (max_val_d - min_val_d)

        print(len(raw_lines))
        dump_to_file(raw_lines,output_name)


def normalize_rewards_m11(read_name, output_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_rewards = []
        normalized_data = []

        for line in csv_obj:
            raw_rewards.append(np.float32(line[9]))
            normalized_data.append(np.array(line))

        min_val = np.min(raw_rewards)
        max_val = np.max(raw_rewards)
        print("this is min val: "+str(min_val))
        print("this is max val: "+str(max_val))
        # zi = 2 * ((xi – xmin) / (xmax – xmin)) – 1
        for line in normalized_data:
            scaled_reward =2.0*(np.float32(line[9]) - min_val) / (max_val - min_val) - 1.0
            line[9] = scaled_reward
            #normalized_data.append(line)
        print(len(normalized_data))
        dump_to_file(normalized_data,output_name)



# doesn't work well -> goes from -5 to 5 or so
def standardize_rewards(read_name, output_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        raw_rewards = []
        normalized_data = []

        for line in csv_obj:
            raw_rewards.append(np.float32(line[9]))
            normalized_data.append(np.array(line))

        mean = np.mean(raw_rewards)
        stddev = np.std(raw_rewards)
        print("this is stddev: "+str(stddev))
        print("this is mean: "+str(mean))

        for line in normalized_data:
            scaled_reward = (np.float32(line[9]) - mean) / (stddev)
            line[9] = scaled_reward

        print(len(normalized_data))
        dump_to_file(normalized_data,output_name)

# doesn't work well -> goes from -5 to 5 or so
def old_to_new(read_name, output_name):
    with open(read_name) as file_obj:
        csv_obj = csv.reader(file_obj)
        new_lines = []

        for line in csv_obj:
            new_line = []
            for i in range(9):
                conv_el = np.float32(line[i])
                if(conv_el == -1.0):
                    conv_el = -2.0
                elif(conv_el == 10):
                    conv_el = -1.0
                elif(conv_el == 20.0):
                    conv_el = 10
                new_line.append(conv_el)
            new_line.append(line[9])
            new_lines.append(np.array(new_line))

        print(len(new_lines))
        dump_to_file(new_lines,output_name)


print("starting data comparison")
old_to_new("rewards3.txt","new_rewards3.txt")
rotate_data("new_rewards3.txt","new_rotated_pts.csv")
remove_duplicates("new_rotated_pts.csv")
normalize_rewards_and_inputs("new_rotated_pts.csv", "new_unique_normalized_rewards_m55.csv", -2, 5)
# normalize_rewards_and_inputs("three_by_three_testdata_m55.txt", "three_by_three_testdata_m55_tran.txt", -1, 4)
# normalize_rewards_m11("unique_rotated_pts.csv","unique_normalized_m11_rewards.csv")
# standardize_rewards("unique_rotated_pts.csv","unique_standardized_rewards.csv")
