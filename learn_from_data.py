#!/usr/bin/env python3
# Tim Marvel
import math
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
neural_net = neural_net_lib.ThreeByThreeSig().to(device)
# TODO try different loss functions
# mse_loss = nn.MSELoss()
l1_loss = nn.SmoothL1Loss()
# TODO try and see what descent method is best
# optimizer = optim.SGD(neural_net.parameters(), lr=0.00003)
optimizer = optim.Adam(neural_net.parameters(), lr=0.0001)

# location of train samples
pos_location, neg_location = tools.get_save_location_three()
print("neg_location")
print(neg_location)
# location of test samples
pos_location_test, neg_location_test = tools.get_save_test_location_three()

list_pos = os.listdir(pos_location)
list_neg = os.listdir(neg_location)

list_pos_test = os.listdir(pos_location_test)
list_neg_test = os.listdir(neg_location_test)

# for each sample there is a state and a reward
neg_size = math.floor(len(list_neg) / 2)
pos_size = math.floor(len(list_pos) / 2)

neg_size_test = math.floor(len(list_neg_test) / 2)
pos_size_test = math.floor(len(list_pos_test) / 2)

print("there are: " + str(pos_size) + " positive training samples")
print("there are: " + str(neg_size) + " negative training samples")

print("there are: " + str(pos_size_test) + " positive test samples")
print("there are: " + str(neg_size_test) + " negative test samples")

list_pos.sort()
list_neg.sort()

list_pos_test.sort()
list_neg_test.sort()


# BUG -> gitignore counts as 0th element
# everything is offset by 1
def get_training_set_pos_three():
    r = random.randint(0, pos_size - 1)
    original = np.load((pos_location + "/" + list_pos[2 * r + 1]))
    reward = np.load((pos_location + "/" + list_pos[2 * r + 2]))
    # a bit of reward shaping
    if reward > 0.5 and original.sum() == 90:
        # nonlocal reward
        reward = 0.25

    elif reward > 0.5 and original.sum() == 6*10-3:
        reward = 0.25

    elif reward > 0.5 and original.sum() == (4 * 10 - 5):
        reward = 0.25

        #print("this is reward 1: " + str(reward))
    '''
    elif reward == 1:
        reward = 2.0
        #print("this is reward 2: " + str(reward))
    '''
    #print("this is reward 3: " + str(reward))
    #exit()

    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


def get_test_set_pos_three():
    r = random.randint(0, pos_size_test - 1)
    original = np.load((pos_location_test + "/" + list_pos_test[2 * r + 1]))
    reward = np.load((pos_location_test + "/" + list_pos_test[2 * r + 2]))

    if reward > 0.5 and original.sum() == 90:
        # nonlocal reward
        reward = 0.25
    elif reward > 0.5 and original.sum() == 6*10-3:
        reward = 0.25
    elif reward > 0.5 and original.sum() == 4*10 - 5:
        reward = 0.25
    '''
    elif reward == 1:
        reward = 2.0
    '''

    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


def get_test_set_neg_three():
    r = random.randint(0, neg_size_test - 1)
    # r = r - r % 2
    original = np.load((neg_location_test + "/" + list_neg_test[2 * r + 1]))
    reward = np.load((neg_location_test + "/" + list_neg_test[2 * r + 2]))
    # reward shaping
    if original[1, 1] == 90:
        # nonlocal reward
        reward = -10.0
    elif original[1, 1] == 0:
        # nonlocal reward
        reward = -10.0
    elif reward == -10:
        reward = -200.0
    elif reward == 0:
        # nonlocal reward
        reward = -0.15

    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


# SAME bug as before with Gitignore file at 0th index
def get_training_set_neg_three():
    # print("negative")
    r = random.randint(0, neg_size - 1)
    # r = r - r % 2
    original = np.load((neg_location + "/" + list_neg[2 * r + 1]))
    # print(original)
    reward = np.load((neg_location + "/" + list_neg[2 * r + 2]))
    # print(reward)
    # reward shaping
    if original[1, 1] == 90:
        # nonlocal reward
        reward = -10.0
    elif original[1, 1] == 0:
        # nonlocal reward
        reward = -10.0
    elif reward == -10:
        reward = -200.0
    elif reward == 0:
        # nonlocal reward
        reward = -0.15

    rot1 = tools.rotate_by_90(original)
    rot2 = tools.rotate_by_90(rot1)
    rot3 = tools.rotate_by_90(rot2)
    return reward, original, rot1, rot2, rot3


def get_random_set_three(is_test):
    toggle = random.random()
    if toggle > 0.4:
        if is_test:
            return get_test_set_pos_three()
        else:
            return get_training_set_pos_three()
    else:
        if is_test:
            return get_test_set_neg_three()
        else:
            return get_training_set_neg_three()


def load_batch_to_torch_three(is_test):
    states = []
    reward = []

    for i in range(0, BATCH_SIZE):
        r, s1, s2, s3, s4 = get_random_set_three(is_test)
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


def evaluate_the_net(current_epoch, evaluation_steps):
    test_losses = []
    backup_graph_name = os.path.abspath(os.path.join(tools.get_working_dir(), '../training_plots/test_result_'
                                                     + str(current_epoch) + '_' + str(evaluation_steps) + '.png'))
    for i in range(0, evaluation_steps):
        loaded_s, loaded_rewards = load_batch_to_torch_three(True)
        result = neural_net.forward(loaded_s)
        loaded_rewards = loaded_rewards.unsqueeze(1)
        test_loss = l1_loss(result, loaded_rewards)
        test_losses.append(test_loss)

    test_losses = np.array(test_losses)
    plt.plot(test_losses)
    plt.ylim(0, 7)
    plt.grid()
    plt.savefig(backup_graph_name)
    plt.clf()


def train_the_net(current_epoch, training_steps):
    train_losses = []

    # set name based on current itt

    backup_net_name = os.path.abspath(os.path.join(tools.get_working_dir(),
                                                   '../saved_nets/neural_net_' + str(current_epoch) + '_' + str(
                                                       training_steps)))
    backup_graph_name = os.path.abspath(os.path.join(tools.get_working_dir(), '../training_plots/train_result_'
                                                     + str(current_epoch) + '_' + str(training_steps) + '.png'))
    print(backup_graph_name)

    for i in range(0, training_steps):
        loaded_s, loaded_rewards = load_batch_to_torch_three(False)
        result = neural_net.forward(loaded_s)
        loaded_rewards = loaded_rewards.unsqueeze(1)
        # TODO investigate different error functions
        # loss = F.smooth_l1_loss(result, loaded_rwds)
        optimizer.zero_grad()
        train_loss = l1_loss(result, loaded_rewards)
        # print("this is train loss")
        # print(train_loss)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss)

    torch.save(neural_net.state_dict(), backup_net_name)
    print("saving results")
    # save this to image based on current_itt
    train_losses = np.array(train_losses)
    plt.plot(train_losses)
    # plt.ylim(0, 5)
    plt.grid()
    plt.savefig(backup_graph_name)
    plt.clf()
    # plt.show()
    # TODO also plot the test losses
