#!/usr/bin/env python
import numpy as npy
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import rankdata

action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT.

action_size = 8
transition_space = 3 
discrete_size = 50
trajectory_length = 40
number_trajectories = 47

max_dist = 5
discrete_size = 50
path_plot = npy.zeros(shape=(discrete_size,discrete_size))
max_path_length=30
current_pose = [0,0]
max_number_demos = 50
trajectory_lengths = npy.zeros(max_number_demos)
state_counter = 0
number_demos = 0
basis_size=3

basis_functions = npy.zeros(shape=(basis_size,discrete_size,discrete_size))

