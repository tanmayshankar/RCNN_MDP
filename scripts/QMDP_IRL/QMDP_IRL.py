#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import rospy
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import random
from scipy.stats import rankdata
from matplotlib.pyplot import *
from scipy import signal
import copy

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
trajectory_index=0
length_index=0

bucket_space = npy.zeros((action_size,transition_space**2))
cummulative = npy.zeros(action_size)
bucket_index = 0
# time_limit = 500

obs_bucket_space = npy.zeros(obs_space**2)
obs_bucket_index =0 
obs_cummulative = 0

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

#### Remember, these are target Q values. We don't need to learn these. 
# q_value_layers = npy.loadtxt(str(sys.argv[2]))
# q_value_layers = q_value_layers.reshape((action_size,discrete_size,discrete_size))

q_value_estimate = npy.ones((action_size,discrete_size,discrete_size))
# q_value_layers /= q_value_layers.sum()

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)

number_trajectories = 6
trajectory_length = 30

trajectories = npy.loadtxt(str(sys.argv[2]))
trajectories = trajectories.reshape((number_trajectories,trajectory_length,2))

observed_trajectories = npy.loadtxt(str(sys.argv[3]))
observed_trajectories = observed_trajectories.reshape((number_trajectories,trajectory_length,2))

actions_taken = npy.loadtxt(str(sys.argv[4]))
actions_taken = actions_taken.reshape((number_trajectories,trajectory_length))
target_actions = npy.zeros(action_size)

def initialize_state():
	global current_pose, from_state_belief

	from_state_belief[24,24]=1.
	current_pose=[24,24]

def initialize_transitions():
	global trans_mat
	#Adding epsilon so that the cummulative distribution has unique values. 
	epsilon=0.001
	trans_mat_1+=epsilon
	trans_mat_2+=epsilon

	trans_mat_1/=trans_mat_1.sum()
 	trans_mat_2/=trans_mat_2.sum()

	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

	print "Transition Matrices:\n",trans_mat

	# for i in range(0,action_size):
	# 	trans_mat[i] = npy.fliplr(trans_mat[i])
	# 	trans_mat[i] = npy.flipud(trans_mat[i])

def initialize_observation():
	global observation_model
	observation_model = npy.array([[0.,0.05,0.],[0.05,0.8,0.05],[0.,0.05,0.]])

	epsilon=0.0001
	observation_model += epsilon
	observation_model /= observation_model.sum()

def display_beliefs():
	global from_state_belief,to_state_belief,target_belief,current_pose

	print "From:"
	for i in range(current_pose[0]-5,current_pose[0]+5):
		print from_state_belief[i,current_pose[1]-5:current_pose[1]+5]
	print "To:"
	for i in range(current_pose[0]-5,current_pose[0]+5):
		print to_state_belief[i,current_pose[1]-5:current_pose[1]+5]
	print "Target:"
	for i in range(current_pose[0]-5,current_pose[0]+5):
		print target_belief[i,current_pose[1]-5:current_pose[1]+5]

def bayes_obs_fusion():
	global to_state_belief, current_pose, observation_model, obs_space, observed_state, corr_to_state_belief
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))
	h = obs_space/2
	for i in range(-h,h+1):
		for j in range(-h,h+1):
			dummy[observed_state[0]+i,observed_state[1]+j] = to_state_belief[observed_state[0]+i,observed_state[1]+j] * observation_model[h+i,h+j]
	corr_to_state_belief[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())	

def initialize_all():
	initialize_state()
	initialize_observation()
	initialize_transitions()
	initialize_unknown_observation()
	initialize_unknown_transitions()
	initialize_model_bucket()
	initialize_obs_model_bucket()

def construct_from_ext_state():
	global from_state_ext, from_state_belief,discrete_size
	d=discrete_size
	from_state_ext[w:d+w,w:d+w] = copy.deepcopy(from_state_belief[:,:])

def belief_prop_extended(action_index):
	global trans_mat_unknown, from_state_ext, to_state_ext, w, discrete_size
	to_state_ext = signal.convolve2d(from_state_ext,trans_mat_unknown[action_index],'same')
	d=discrete_size
	##NOW MUST FOLD THINGS:
	for i in range(0,2*w):
		to_state_ext[i+1,:]+=to_state_ext[i,:]
		to_state_ext[i,:]=0
		to_state_ext[:,i+1]+=to_state_ext[:,i]
		to_state_ext[:,i]=0
		to_state_ext[d+2*w-i-2,:]+= to_state_ext[d+2*w-i-1,:]
		to_state_ext[d+2*w-i-1,:]=0
		to_state_ext[:,d+2*w-i-2]+= to_state_ext[:,d+2*w-i-1]
		to_state_ext[:,d+2*w-i-1]=0

	to_state_belief[:,:] = copy.deepcopy(to_state_ext[w:d+w,w:d+w])

def feedforward_recurrence():
	global from_state_belief, to_state_belief
	from_state_belief = copy.deepcopy(to_state_belief)

def calc_softmax():
	global qmdp_values, qmdp_values_softmax

	for act in range(0,action_size):
		qmdp_values_softmax[act] = npy.exp(qmdp_values[act]) / npy.sum(npy.exp(qmdp_values), axis=0)

def update_QMDP_values():
	global to_state_belief, q_value_estimate, qmdp_values

	for act in range(0,action_size):
		qmdp_values[act] = npy.sum(q_value_estimate[act]*to_state_belief)

def IRL_backprop(action_index):
	global to_state_belief, q_value_estimate, qmdp_values_softmax, trajectory_index, length_index, target_actions

	update_QMDP_values()
	calc_softmax()

	for trajectory_index in range(0,number_trajectories):
		for length_index in range(0,trajectory_length):
			for act in range(0,action_size):
				q_value_estimate[act,:,:] -= alpha*(qmdp_values_softmax[act]-target_actions[act])*to_state_belief[:,:]

def parse_data():
	global observed_state, trajectory_index, length_index, target_actions

	observed_state[:] = observed_trajectories[trajectory_index,length_index,:]
	target_actions[:] = 0
	target_actions[actions_taken[trajectory_index,length_index]] = 1


def master(action_index, time_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose
	global trajectory_index, length_index

	construct_from_ext_state()
	belief_prop_extended(actions_taken[trajectory_index,length_index])
	bayes_obs_fusion()


	
	feedforward_recurrence()	

initialize_all()
