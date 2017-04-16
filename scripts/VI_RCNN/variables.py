#!/usr/bin/env python

import numpy as npy
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.stats import rankdata
from matplotlib.pyplot import *
from scipy import signal
import copy

basis_size = 3
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........

transition_space = 3

reward_function = npy.loadtxt(str(sys.argv[1]))
trans_mat = npy.loadtxt(str(sys.argv[2]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))	

value_function = npy.zeros(shape=(discrete_size,discrete_size))
optimal_policy = npy.zeros(shape=(discrete_size,discrete_size))

gamma = 0.95
time_limit = 100

action_reward_function = npy.zeros((action_size,discrete_size,discrete_size))
action_value_layers  = npy.zeros(shape=(action_size,discrete_size,discrete_size))
q_value_layers  = npy.zeros(shape=(action_size,discrete_size,discrete_size))
