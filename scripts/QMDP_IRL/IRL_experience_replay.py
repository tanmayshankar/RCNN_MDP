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
from random import shuffle

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
current_pose = npy.zeros(2)
current_pose = current_pose.astype(int)

state_counter = 0
action = 'w'

#### Take required inputs. 
trans_mat = npy.loadtxt(str(sys.argv[1]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

print trans_mat
#### Remember, these are target Q values. We don't need to learn these. 
# q_value_layers = npy.loadtxt(str(sys.argv[2]))
# q_value_layers = q_value_layers.reshape((action_size,discrete_size,discrete_size))

q_value_estimate = npy.ones((action_size,discrete_size,discrete_size))
# q_value_layers /= q_value_layers.sum()

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)

number_trajectories =47
trajectory_length = 30

trajectories = npy.loadtxt(str(sys.argv[2]))
trajectories = trajectories.reshape((number_trajectories,trajectory_length,2))

observed_trajectories = npy.loadtxt(str(sys.argv[3]))
observed_trajectories = observed_trajectories.reshape((number_trajectories,trajectory_length,2))

actions_taken = npy.loadtxt(str(sys.argv[4]))
actions_taken = actions_taken.reshape((number_trajectories,trajectory_length))
target_actions = npy.zeros(action_size)

time_limit = number_trajectories*trajectory_length
learning_rate = 1
annealing_rate = learning_rate/5

def initialize_state():
	# global current_pose, from_state_belief, observed_state
	global observed_state
	from_state_belief[observed_state[0],observed_state[1]] = 1.

def initialize_observation():
	global observation_model
	# observation_model = npy.array([[0.05,0.05,0.05],[0.05,0.6,0.05],[0.05,0.05,0.05]])
	# observation_model = npy.array([[0.05,0.05,0.05],[0.05,0.6,0.05],[0.05,0.05,0.05]])
	observation_model = npy.array([[0.05,0.1,0.05],[0.1,0.4,0.1],[0.05,0.1,0.05]])

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

def construct_from_ext_state():
	global from_state_ext, from_state_belief,discrete_size
	d=discrete_size
	from_state_ext[w:d+w,w:d+w] = copy.deepcopy(from_state_belief[:,:])

def belief_prop_extended(action_index):
	global trans_mat, from_state_ext, to_state_ext, w, discrete_size
	to_state_ext = signal.convolve2d(from_state_ext,trans_mat[action_index],'same')
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
	global from_state_belief, to_state_belief, corr_to_state_belief
	# from_state_belief = copy.deepcopy(corr_to_state_belief)
	from_state_belief = copy.deepcopy(to_state_belief)

def calc_softmax():
	global qmdp_values, qmdp_values_softmax

	for act in range(0,action_size):
		qmdp_values_softmax[act] = npy.exp(qmdp_values[act]) / npy.sum(npy.exp(qmdp_values), axis=0)

def dummy_softmax():
	global qmdp_values, qmdp_values_softmax, action_size
	
	# for act in range(0,action_size):
	qmdp_values_softmax = npy.zeros(action_size)
	qmdp_values_softmax[npy.argmax(qmdp_values)]=1.

def update_QMDP_values():
	global to_state_belief, q_value_estimate, qmdp_values, from_state_belief

	for act in range(0,action_size):
		# qmdp_values[act] = npy.sum(q_value_estimate[act]*to_state_belief)
		qmdp_values[act] = npy.sum(q_value_estimate[act]*from_state_belief)

# def IRL_backprop():
def Q_backprop():
	global to_state_belief, q_value_estimate, qmdp_values_softmax, learning_rate, annealing_rate
	global trajectory_index, length_index, target_actions, time_index

	update_QMDP_values()
	calc_softmax()
	# dummy_softmax()
 
	alpha = learning_rate - annealing_rate * time_index

	for act in range(0,action_size):
		q_value_estimate[act,:,:] = q_value_estimate[act,:,:] - alpha*(qmdp_values_softmax[act]-target_actions[act])*from_state_belief[:,:]

		# print "Ello", alpha*(qmdp_values_softmax[act]-target_actions[act])*from_state_belief[:,:]
def parse_data():
	global observed_state, trajectory_index, length_index, target_actions, current_pose, trajectories

	observed_state[:] = observed_trajectories[trajectory_index,length_index,:]
	target_actions[:] = 0
	target_actions[actions_taken[trajectory_index,length_index]] = 1
	current_pose[:] = trajectories[trajectory_index,length_index,:]

def master():
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose
	global trajectory_index, length_index

	construct_from_ext_state()
	belief_prop_extended(actions_taken[trajectory_index,length_index])
	
	print observed_state, current_pose, target_actions, qmdp_values_softmax
	# bayes_obs_fusion()
	parse_data()

	Q_backprop()
	# display_beliefs()
	feedforward_recurrence()	

def Inverse_Q_Learning():
	global trajectories, trajectory_index, length_index, trajectory_length, number_trajectories, time_index
	time_index = 0
	for trajectory_index in range(0,number_trajectories):
		for length_index in range(0,trajectory_length):			
			if (from_state_belief.sum()>0):
				master()
				time_index += 1
				print time_index
			else: 
				print "We've got a problem"

trajectory_pick = range(0,number_trajectories-1)
print trajectory_pick
shuffle(trajectory_pick)
print trajectory_pick

def experience_replay():
	global trajectories, trajectory_index, length_index, trajectory_length, number_trajectories, time_index

	time_index = 0

	# for trajectory_index in range(0,number_trajectories):
	# for trajectory_index in trajectory_pick:
	for i in range(0,number_trajectories-1):
		trajectory_index = trajectory_pick[i]
		initialize_all()
		for length_index in range(0,trajectory_length):			
			if (from_state_belief.sum()>0):
				master()
				time_index += 1
				print time_index

trajectory_index = 0
length_index = 0
parse_data()



experience_replay()

# for i in range(0,action_size):
# 	print "New action."
# 	for j in range(0,discrete_size):
# 		print q_value_estimate[i,j,:]


value_function = npy.amax(q_value_estimate, axis=0)
imshow(value_function, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show()


with file('Q_Value_Estimate.txt','w') as outfile:
	for data_slice in q_value_estimate:
		outfile.write('#Q_Value_Estimate.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')