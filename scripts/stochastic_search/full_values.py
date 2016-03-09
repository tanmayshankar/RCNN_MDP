#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import rospy
# from std_msgs.msg import String
# import roslib
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import random
from scipy.stats import rankdata
from matplotlib.pyplot import *
from scipy import signal
import copy

basis_size = 3
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 8
# action_space = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........

#Transition space size determines size of convolutional filters. 
transition_space = 3

# basis_functions = npy.loadtxt(str(sys.argv[1]))
# reward_weights = npy.loadtxt(str(sys.argv[2]))
# basis_functions = basis_functions.reshape((basis_size,discrete_size,discrete_size))

# reward_function = basis_functions[0]*reward_weights[0]+basis_functions[2]*reward_weights[2]+basis_functions[1]*reward_weights[1]

#Static / instantaneous reward. 
# reward_function = npy.loadtxt(str(sys.argv[1]))
# reward_function = basis_functions[0,:,:]
# reward_function /=1000.0 

time_limit = 100

npy.set_printoptions(precision=3)

value_functions = npy.zeros(shape=(time_limit,discrete_size,discrete_size))
value_function = npy.zeros(shape=(discrete_size,discrete_size))

optimal_policy = npy.zeros(shape=(discrete_size,discrete_size))

gamma = 0.95
# gamma = 1.

trans_mat = npy.zeros(shape=(action_size,transition_space,transition_space))

def conv_transition_filters():
	global trans_mat
	# trans_mat_1 = [[0.,0.97,0.],[0.01,0.01,0.01],[0.,0.,0.]]
	# trans_mat_2 = [[0.97,0.01,0.],[0.01,0.01,0.],[0.,0.,0.]]

	trans_mat_1 = [[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]]
	trans_mat_2 = [[0.7,0.1,0.],[0.1,0.1,0.],[0.,0.,0.]]
	
	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

	# for i in range(0,action_size):
	# 	trans_mat[i] = npy.fliplr(trans_mat[i])
	# 	trans_mat[i] = npy.flipud(trans_mat[i])


conv_transition_filters()

print "Transition Matrices:\n",trans_mat

# print "\nHere's the reward.\n"
# for i in range(0,discrete_size):
# 	print reward_function[i]

to_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
from_state_belief = npy.zeros(shape=(discrete_size,discrete_size))
target_belief = npy.zeros(shape=(discrete_size,discrete_size))

from_state_belief[24,24]=0.8
from_state_belief[25,24]=0.2

current_pose=[24,24]

trans_mat_unknown = npy.zeros(shape=(action_size,transition_space,transition_space))

# def reset_belief():
# 	global current_pose
# 	max_val_location = npy.unravel_index(npy.argmax(from_state_belief),from_state_belief.shape)
# 	print "Reset:",max_val_location
# 	from_state_belief[:,:]=0.
# 	from_state_belief[max_val_location[0],max_val_location[1]]=1.

def initialize_unknown_transitions():
	global trans_mat_unknown

	for i in range(0,transition_space):
		for j in range(0,transition_space):
	# 		trans_mat_unknown[:,i,j] = random.random()
			trans_mat_unknown[:,i,j] = 1.
	for i in range(0,action_size):
		trans_mat_unknown[i,:,:] /=trans_mat_unknown[i,:,:].sum()

initialize_unknown_transitions()

obs_space = 3
observation_model = npy.zeros(shape=(obs_space,obs_space))

def initialize_observation():
	# print "bleh"
	global observation_model
	observation_model = npy.array([[0.,0.05,0.],[0.05,0.8,0.05],[0.,0.05,0.]])
	print observation_model

initialize_observation()

def fuse_observations():
	global from_state_belief
	global current_pose
	global observation_model

	dummy = npy.zeros(shape=(discrete_size,discrete_size))

	l = obs_space/2
	for i in range(-l,l+1):
		for j in range(-l,l+1):
			dummy[i+current_pose[0],j+current_pose[1]] = from_state_belief[i+current_pose[0],j+current_pose[1]]*observation_model[l+i,l+j]

	from_state_belief[:,:] = dummy[:,:]/dummy.sum()

# observation_model()

def calculate_target(action_index):
	# global trans_mat_unknown
	# global to_state_belief
	# global from_state_belief
	global target_belief

	#TARGET TYPE 1 
	# target_belief[:,:]=0.
	# target_belief[to_state[0],to_state[1]]=1.

	#TARGET TYPE 2: actual_T * from_belief
	target_belief = from_state_belief
	target_belief = signal.convolve2d(from_state_belief,trans_mat[action_index],'same','fill',0)
	
	#TARGET TYPE 3: 
	# target_belief = from_state_belief
	# target_belief = signal.convolve2d(from_state_belief,trans_mat[action_index],'same','fill',0)
	#Fuse with Observations

	#TARGET TYPE 4: 
	

	# if (target_belief.sum()<1.):
	# 	target_belief /= target_belief.sum(

def belief_prop(action_index):
	global trans_mat_unknown
	global to_state_belief
	global from_state_belief
	# global target_belief

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()
	# from_state_belief = to_state_belief

def full_value(action_index):
	global trans_mat_unknown
	global to_state_belief
	global from_state_belief
	global target_belief
	alpha = 0.01

	w = transition_space/2
	print "W:",w
	# for ai in range(-transition_space/2,transition_space/2+1):
		# for aj in range(-transition_space/2,transition_space/2+1):

	print "From:"
	for i in range(20,30):
		print from_state_belief[i,20:30]
	# for i in range(0,50):
	# 	print from_state_belief[i,0:50]
	print "To:"
	for i in range(20,30):
		print to_state_belief[i,20:30]
	# for i in range(0,50):
	# 	print to_state_belief[i,0:50]
	print "Target:",
	for i in range(20,30):
		print target_belief[i,20:30]
	# for i in range(0,50):	
	# 	print target_belief[i,0:50]

	trans_linspace_size = 21
	trans_val_space = npy.linspace(0,1,trans_linspace_size)

	loss = npy.zeros(trans_linspace_size)

	new_trans_mat = npy.zeros(shape=(transition_space,transition_space))

	dummy_max = 10000.

	for ti in range(0,transition_space):
		for tj in range(0,transition_space):
			# for v in range(npy.linspace())
			# for v in trans_val_space:

			loss[:] = 0.
			dummy_max = 10000.
			
			for vi in range(0,trans_linspace_size):
				for i in range(0,discrete_size):
					for j in range(0,discrete_size):
						# loss[vi] += (target_belief[i,j]-to_state_belief[i,j])**2

						loss[vi] += (target_belief[i,j]-to_state_belief[i,j])**2+

			if (loss[vi]<dummy_max):
				dummy_max = loss[vi]
				new_trans_mat[ti,tj]=trans_val_space[vi]

			trans_mat_unknown[ti,tj] = (1-alpha)*trans_mat_unknown[ti,tj]+alpha*new_trans_mat[ti,tj]

	# for ai in range(-w,w+1):
	# 	for aj in range(-w,w+1):
	# 		for i in range(0,discrete_size-2):
	# 			for j in range(0,discrete_size-2):

	# 				loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj])

	# 		trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
	# 		if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
	# 			trans_mat_unknown[action_index,w+ai,w+aj]=0
	# 		trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()



def back_prop(action_index):
	global trans_mat_unknown
	global to_state_belief
	global from_state_belief
	global target_belief

	loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.01

	w = transition_space/2
	print "W:",w
	# for ai in range(-transition_space/2,transition_space/2+1):
		# for aj in range(-transition_space/2,transition_space/2+1):

	print "From:"
	for i in range(20,30):
		print from_state_belief[i,20:30]
	# for i in range(0,50):
	# 	print from_state_belief[i,0:50]
	print "To:"
	for i in range(20,30):
		print to_state_belief[i,20:30]
	# for i in range(0,50):
	# 	print to_state_belief[i,0:50]
	print "Target:",
	for i in range(20,30):
		print target_belief[i,20:30]
	# for i in range(0,50):	
	# 	print target_belief[i,0:50]

	for ai in range(-w,w+1):
		for aj in range(-w,w+1):
			for i in range(0,discrete_size-2):
				for j in range(0,discrete_size-2):

					loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj])

			trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
			if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
				trans_mat_unknown[action_index,w+ai,w+aj]=0
			trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()

def master(action_index):
	global trans_mat_unknown
	global to_state_belief
	global from_state_belief
	global target_belief
	global current_pose

	# if (random.random()>0.):
	# 	reset_belief()

	belief_prop(action_index)
	calculate_target(action_index)
	# back_prop(action_index)
	full_value(action_index)
	
	# from_state_belief = to_state_belief
	##### IN THE ALTERNATE GRAPH, WE UPDATE FROM_STATE_BELIEF AS TARGET_BELIEF

	from_state_belief = target_belief
	fuse_observations()
	# back_prop(action_index)

	print "current_pose:",current_pose
	print "Transition Matrix: ",action_index,"\n"
	print npy.flipud(npy.fliplr(trans_mat_unknown[action_index,:,:]))

state_counter = 0
action = 'w'

print trans_mat_unknown
while (action!='q'):		
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........
		# 
		action = raw_input("Hit a key now: ")
		if action=='w':			
			state_counter+=1	
			# current_demo.append([current_pose[0]+1,current_pose[1]])
			current_pose[0]+=1			
			action_index=0

		if action=='a':			
			state_counter+=1		
			# current_demo.append([current_pose[0],current_pose[1]-1])
			current_pose[1]-=1			
			action_index=2

		if action=='d':			
			state_counter+=1
			# current_demo.append([current_pose[0],current_pose[1]+1])
			current_pose[1]+=1
			action_index=1

		if action=='s':			
			state_counter+=1
			# current_demo.append([current_pose[0]-1,current_pose[1]])
			current_pose[0]-=1
			action_index=3

		if ((action=='wa')or(action=='aw')):			
			state_counter+=1	
			# current_demo.append([current_pose[0]+1,current_pose[1]-1])
			current_pose[0]+=1	
			current_pose[1]-=1						
			action_index=4

		if ((action=='sa')or(action=='as')):					
			state_counter+=1		
			# current_demo.append([current_pose[0]-1,current_pose[1]-1])
			current_pose[1]-=1
			current_pose[0]-=1		
			action_index=6

		if ((action=='sd')or(action=='ds')):					
			state_counter+=1
			# current_demo.append([current_pose[0]-1,current_pose[1]+1])
			current_pose[1]+=1
			current_pose[0]-=1
			action_index=7

		if ((action=='wd')or(action=='dw')):					
			state_counter+=1
			# current_demo.append([current_pose[0]+1,current_pose[1]+1])
			current_pose[0]+=1			
			current_pose[1]+=1			
			action_index=5

		# path_plot[current_pose[0]][current_pose[1]]=1				
		master(action_index)
	









def conv_layer():	
	global value_function
	global trans_mat
	action_value_layers = npy.zeros(shape=(action_size,discrete_size,discrete_size))
	layer_value = npy.zeros(shape=(discrete_size,discrete_size))
	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		action_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Max pooling over actions. 
	value_function = gamma*npy.amax(action_value_layers,axis=0)
	# layer_value = gamma*npy.amax(action_value_layers,axis=0)
	print "The next value function.",value_function
	optimal_policy[:,:] = npy.argmax(action_value_layers,axis=0)
	# return layer_value

def reward_bias():
	global value_function
	value_function = value_function + reward_function

def recurrent_value_iteration():
	global value_function
	print "Start iterations."
	t=0	
	while (t<time_limit):
		conv_layer()
		reward_bias()		
		t+=1
		print t
	
# recurrent_value_iteration()

# print "Here's the policy."
# for i in range(0,discrete_size):
# 	print optimal_policy[i]

# policy_iteration()

# print "These are the value functions."
# for t in range(0,time_limit):
# 	print value_functions[t]

# with file('reward_function.txt','w') as outfile: 
# 	outfile.write('#Reward Function.\n')
# 	npy.savetxt(outfile,1000*reward_function,fmt='%-7.2f')

# with file('output_policy.txt','w') as outfile: 
# 	outfile.write('#Policy.\n')
# 	npy.savetxt(outfile,optimal_policy,fmt='%-7.2f')

# with file('value_function.txt','w') as outfile: 
# 	outfile.write('#Value Function.\n')
# 	npy.savetxt(outfile,1000*value_function,fmt='%-7.2f')