#!/usr/bin/env python
import numpy as npy
from variables import *

action_size = 8

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
		initialize_all()
		for length_index in range(0,trajectory_length):			
			if (from_state_belief.sum()>0):
				master()
				time_index += 1
				print time_index
			else: 
				print "We've got a problem"

trajectory_index = 0
length_index = 0
parse_data()

Inverse_Q_Learning()

value_function = npy.amax(q_value_estimate, axis=0)
plt.imshow(value_function, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
plt.colorbar()
plt.show()

with file('Q_Value_Estimate.txt','w') as outfile:
	for data_slice in q_value_estimate:
		outfile.write('#Q_Value_Estimate.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')