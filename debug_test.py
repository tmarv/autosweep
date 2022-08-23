#!/usr/bin/env python3
# Tim Marvel
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import src.tools
from src import reward_manager, tools

'''
filename3 = tools.get_text_file_names()[0]
print(filename3)
dataPoints = np.array(pd.read_csv(filename3, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(len(dataPoints))

_rewards3_text_file = "replacement.txt"
_rewards3_text_replacement = open(_rewards3_text_file, 'w')

for i in range(len(dataPoints)):
    if dataPoints[i][9] == -10:
        dataPoints[i][9] = -64
    elif dataPoints[i][9] == 0:
        dataPoints[i][9] = -0.15
    inputs_list = dataPoints[i].flatten().tolist()
    list = ','.join(str(v) for v in inputs_list)
    _rewards3_text_replacement.write(list+"\n")

_rewards3_text_replacement.close()

'''

filename5 = tools.get_text_file_names()[1]
print(filename5)
dataPoints = np.array(pd.read_csv(filename5, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))
print(len(dataPoints))

_rewards5_text_file = "replacement_5.txt"
_rewards5_text_replacement = open(_rewards5_text_file, 'w')

for i in range(len(dataPoints)):
    if dataPoints[i][25] == -10:
        dataPoints[i][25] = -64
    elif dataPoints[i][25] == 0:
        dataPoints[i][25] = -0.25

    inputs_list = dataPoints[i].flatten().tolist()
    list = ','.join(str(v) for v in inputs_list)
    _rewards5_text_replacement.write(list+"\n")

_rewards5_text_replacement.close()

print("finished")
