#!/usr/bin/env python3
# Tim Marvel
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.nn as nn

# self made modules
from src import neural_net_lib
from src import tools

# Global variables
BATCH_SIZE = 128
steps_done = 0
TRAINING_STPS = 2000
TRAIN_ITER = 0
NAME = 'nets/neural_net_' + str(TRAINING_STPS)
# prepare ml stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neural_net = neural_net_lib.ThreeByThreeSig()
# TODO try different loss functions
# mse_loss = nn.MSELoss()
l1_loss = nn.SmoothL1Loss()
# TODO try and see what descent method is best
# optimizer = optim.SGD(neural_net.parameters(), lr=0.00003)
optimizer = optim.Adam(neural_net.parameters(), lr=0.00007)

# location of train samples
pos_location, neg_location = tools.get_save_location()
# location of test samples
test_pos_location, test_pos_location = tools.get_save_test_location()

list_pos = os.listdir(pos_location)
list_neg = os.listdir(neg_location)

# for each sample there is a state and a reward
neg_size = len(list_neg) / 2
pos_size = len(list_pos) / 2

print("there are: " + str(pos_size)+" positive training samples")
print("there are: " + str(neg_size)+" negative training samples")

list_pos.sort()
list_neg.sort()


def get_training_set_pos():
    r = random.randint(0, pos_size)
    r = r - r % 2
    original = np.load((pos_location + list_pos[r]))
    reward = np.load((pos_location + list_pos[r + 1]))

    # a bit of reward shaping helps
    if reward > 4:
        nonlocal reward
        reward = 1

    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


def get_test_set_pos():
    print("hello")


def get_training_set_neg():
    r = random.randint(0, neg_size)
    r = r - r % 2
    original = np.load((neg_location + list_neg[r]))
    reward = np.load((neg_location + list_neg[r + 1]))
    # reward shaping
    if original[1, 1] == 90:
        nonlocal reward
        reward = -10.0
    elif original[1, 1] == 0:
        nonlocal reward
        reward = -10.0
    else:
        nonlocal reward
        reward = -2.0
    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


def get_random_set():
    toggle = random.random()
    if toggle > 0.5:
        return get_training_set_pos()
    else:
        return get_training_set_neg()


def load_batch_to_torch():
    states = []
    reward = []

    for i in range(0, BATCH_SIZE):
        r, s1, s2, s3, s4 = get_random_set()
        reward.append(r)
        reward.append(r)
        reward.append(r)
        reward.append(r)
        states.append(s1)
        states.append(s2)
        states.append(s3)
        states.append(s4)

    reward = np.array(reward)
    loaded_states = torch.cat([torch.tensor(states)]).to(device).to(dtype=torch.float)
    loaded_rewards = torch.cat([torch.tensor(reward)]).to(device).to(dtype=torch.float)
    return loaded_states, loaded_rewards


def train_the_net(current_itt, training_steps):
    train_losses = []
    test_losses = []
    # set name based on current itt
    backup_net_name = 'nets/neural_net_' + str(current_itt) + '_' + str(training_steps)
    backup_graph_name = 'train_result_' + str(current_itt) + '_' + str(training_steps)
    for i in range(0, training_steps):
        loaded_s, loaded_rewards = load_batch_to_torch()
        result = neural_net.forward(loaded_s)
        loaded_rewards = loaded_rewards.unsqueeze(1)
        # TODO investigate different error functions
        # loss = F.smooth_l1_loss(result, loaded_rwds)
        optimizer.zero_grad()
        train_loss = l1_loss(result, loaded_rewards)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss)

    torch.save(neural_net.state_dict(), backup_net_name)
    print("saving results")
    # save this to image based on current_itt
    train_losses = np.array(train_losses)
    fig = plt.plot(train_losses)
    # TODO also plot the test losses
    fig.savefig(backup_graph_name)
    # plt.show()


