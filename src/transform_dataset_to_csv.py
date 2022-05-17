import os
import math
import numpy as np
import csv

import tools

pos_location, neg_location = tools.get_save_location_three()

list_pos = os.listdir(pos_location)
list_neg = os.listdir(neg_location)

list_pos.sort()
list_neg.sort()

neg_size = math.floor(len(list_neg) / 2)
pos_size = math.floor(len(list_pos) / 2)

csv_file_name = "gridAndRewards.csv"


csv_file = open(csv_file_name, 'w')
csvwriter = csv.writer(csv_file)
header = ['11','12','13','21','22','23','31','32','33','reward']
csvwriter.writerow(header)

for i in range(0, neg_size-1):
    state = np.load(neg_location+"/"+list_neg[2*i+1]).tolist()
    reward = np.load(neg_location+"/"+list_neg[2*i+2]).tolist()
    reward = [reward]
    to_print = (state[0]+state[1]+state[2]+reward)
    csvwriter.writerow(to_print)


for i in range(0, pos_size-1):
    state = np.load(pos_location+"/"+list_pos[2*i+1]).tolist()
    reward = np.load(pos_location+"/"+list_pos[2*i+2]).tolist()
    reward = [reward]
    to_print = (state[0]+state[1]+state[2]+reward)
    csvwriter.writerow(to_print)

'''
for i in range (0, pos_size-1):
    print(list_pos[2*i+1])
    print(list_pos[2*i+2])
'''