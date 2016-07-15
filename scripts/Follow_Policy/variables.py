#!/usr/bin/env python
import numpy as npy
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import sys
import random
import copy

discrete_size = 50
max_path_length=40
current_pose = [0,0]
max_number_demos = 70
trajectory_lengths = npy.zeros(max_number_demos)

state_counter = 0
number_demos = 0

action_size=9
transition_space = 3
trajectories = [[[0,0],[1,2],[3,4]]]
observed_trajectories = [[[0,0],[1,2],[3,4]]]
actions_taken = [[0,0]]

trans_mat_1 = npy.loadtxt(str(sys.argv[3]))
trans_mat_1 = trans_mat_1.reshape((action_size-1,transition_space,transition_space))
trans_mat = npy.zeros((action_size,transition_space,transition_space))

for i in range(0,action_size-1):
	trans_mat[i] = trans_mat_1[i]
trans_mat[action_size-1,1,1]=1.

optimal_policy = npy.loadtxt(str(sys.argv[1]))
optimal_policy = optimal_policy.astype(int)

reward_function = npy.loadtxt(str(sys.argv[2]))
max_val = npy.amax(reward_function)
max_val_location = npy.unravel_index(npy.argmax(reward_function),reward_function.shape)

## ACTION SPACE:
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])

bucket_space = npy.zeros((action_size,transition_space**2))
cummulative = npy.zeros(action_size)
bucket_index = 0

obs_space=3
observation_model = npy.zeros(shape=(obs_space,obs_space))
observed_state = npy.zeros(2)

obs_bucket_space = npy.zeros(obs_space**2)
obs_bucket_index =0 
obs_cummulative = 0
npy.set_printoptions(precision=3)











###### DEFINITIONS

basis_size = 3
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

#Transition space size determines size of convolutional filters. 
transition_space = 3
obs_space = 3
h=obs_space/2
time_limit = 500

bucket_space = npy.zeros((action_size,transition_space**2))
cummulative = npy.zeros(action_size)
bucket_index = 0
# time_limit = 500

obs_bucket_space = npy.zeros(obs_space**2)
obs_bucket_index =0 
obs_cummulative = 0
observed_state = npy.zeros(2)

npy.set_printoptions(precision=3)

value_function = npy.zeros(shape=(discrete_size,discrete_size))
optimal_policy = npy.zeros(shape=(discrete_size,discrete_size))

#### DEFINING DISCOUNT FACTOR
gamma = 0.95
# gamma = 1.

#### DEFINING TRANSITION RELATED VARIABLES
trans_mat = npy.zeros(shape=(action_size,transition_space,transition_space))
trans_mat_unknown = npy.zeros(shape=(action_size,transition_space,transition_space))

#### DEFINING STATE BELIEF VARIABLES
to_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
from_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
target_belief = npy.zeros(shape=(discrete_size,discrete_size))
corr_to_state_belief = npy.zeros((discrete_size,discrete_size))
#### DEFINING EXTENDED STATE BELIEFS 
w = transition_space/2
to_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))
from_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))

#### DEFINING OBSERVATION RELATED VARIABLES
observation_model = npy.zeros(shape=(obs_space,obs_space))
obs_model_unknown = npy.ones(shape=(obs_space,obs_space))
observed_state = npy.zeros(2)

#### DEFINING QMDP RELATED VARIABLES
q_value_layers = npy.zeros((action_size,discrete_size,discrete_size))	
qmdp_values = npy.zeros(action_size)
take_action = 0

state_counter = 0
action = 'w'

learning_rate = 0.05
annealing_rate = (learning_rate/5)/time_limit
learning_rate_obs = 0.01
annealing_rate_obs = (learning_rate/5)/time_limit

norm_sum_bel=0.

#### Take required inputs. 
trans_mat = npy.loadtxt(str(sys.argv[1]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

q_value_layers = npy.loadtxt(str(sys.argv[2]))
q_value_layers = q_value_layers.reshape((action_size,discrete_size,discrete_size))

reward_function = npy.loadtxt(str(sys.argv[3]))

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)

max_val_location = npy.unravel_index(npy.argmax(reward_function),reward_function.shape)
path_plot = copy.deepcopy(reward_function)
max_val = npy.amax(path_plot)