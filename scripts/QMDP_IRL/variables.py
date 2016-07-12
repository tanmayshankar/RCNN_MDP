#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy import signal
import copy

###### DEFINITIONS
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 9
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT, NOTHING.

#Transition space size determines size of convolutional filters. 
transition_space = 3
obs_space = 3
h=obs_space/2
trajectory_index=0
length_index=0

npy.set_printoptions(precision=3)

#### DEFINING STATE BELIEF VARIABLES
to_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
from_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
corr_to_state_belief = npy.zeros((discrete_size,discrete_size))
backprop_belief = npy.zeros((discrete_size,discrete_size))

#### DEFINING EXTENDED STATE BELIEFS 
w = transition_space/2
to_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))
from_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))

#### DEFINING OBSERVATION RELATED VARIABLES
observation_model = npy.zeros(shape=(obs_space,obs_space))
observed_state = npy.zeros(2)
current_pose = npy.zeros(2)
current_pose = current_pose.astype(int)
observed_state = observed_state.astype(int)

#### Take required inputs. 
trans_mat_1 = npy.loadtxt(str(sys.argv[1]))
trans_mat_1 = trans_mat_1.reshape((action_size-1,transition_space,transition_space))
trans_mat = npy.zeros((action_size,transition_space,transition_space))

for i in range(0,8):
	trans_mat[i]=trans_mat_1[i]
trans_mat[8,1,1]=1.
print trans_mat

q_value_estimate = npy.ones((action_size,discrete_size,discrete_size))
reward_estimate = npy.zeros((action_size,discrete_size,discrete_size))
q_value_layers = npy.zeros((action_size,discrete_size,discrete_size))
value_function = npy.zeros((discrete_size,discrete_size))

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)

number_trajectories = 97
trajectory_length = 30

trajectories = npy.loadtxt(str(sys.argv[2]))
trajectories = trajectories.reshape((number_trajectories,trajectory_length,2))

observed_trajectories = npy.loadtxt(str(sys.argv[3]))
observed_trajectories = observed_trajectories.reshape((number_trajectories,trajectory_length,2))

actions_taken = npy.loadtxt(str(sys.argv[4]))
actions_taken = actions_taken.reshape((number_trajectories,trajectory_length))
target_actions = npy.zeros(action_size)
belief_target_actions = npy.zeros(action_size)

time_limit = number_trajectories*trajectory_length
learning_rate = 0.5
annealing_rate = (learning_rate/5)/time_limit

from_belief_vector = npy.zeros((trajectory_length, discrete_size, discrete_size))
reward_gradients = npy.zeros((action_size,discrete_size,discrete_size))
reward_grad_accum = npy.zeros((action_size,discrete_size,discrete_size))
epsilon=0.0001
reward_grad_temp = epsilon*npy.ones((discrete_size,discrete_size))
# reward_grad_temp = npy.zeros((discrete_size,discrete_size))
rms_decay = 0.9