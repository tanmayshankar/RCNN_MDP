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


# import cProfile
# cProfile.run('foo()')


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
reward_function = npy.loadtxt(str(sys.argv[1]))
# reward_function = basis_functions[0,:,:]
# reward_function /=1000.0
# 
time_limit = 100

reward_function += npy.amin(reward_function)
reward_function /= npy.amax(reward_function)
# reward_function -= npy.amax(reward_function)/5

action_factor_reward = 0.05

value_functions = npy.zeros(shape=(time_limit,discrete_size,discrete_size))
value_function = npy.zeros(shape=(discrete_size,discrete_size))

optimal_policy = npy.zeros(shape=(discrete_size,discrete_size))

gamma = 0.95
# gamma = 0.90
# gamma = 1.

# trans_mat = npy.zeros(shape=(action_size,transition_space,transition_space))
trans_mat = npy.loadtxt(str(sys.argv[2]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

action_reward_function = npy.zeros((action_size,discrete_size,discrete_size))
action_value_layers = npy.zeros(shape=(action_size,discrete_size,discrete_size))

def modify_trans_mat():
	global trans_mat
	epsilon = 0.0001
	for i in range(0,action_size):
		trans_mat[i][:][:] += epsilon
		trans_mat[i] /= trans_mat[i].sum()

	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])
		
def create_action_reward():
	global reward_function, action_reward_function, action_factor_reward

	for i in range(0,action_size):
		action_reward_function[i,:,:] = copy.deepcopy(reward_function) - (i%4)*action_factor_reward * npy.amax(reward_function)

def initialize():
	modify_trans_mat()
	create_action_reward()

initialize()

def conv_transition_filters():
	global trans_mat
	trans_mat_1 = [[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]]
	# trans_mat_2 = [[0.7,0.1,0.],[0.1,0.1,0.],[0.,0.,0.]]
	trans_mat_2 = [[0.55,0.15,0.],[0.15,0.15,0.],[0.,0.,0.]]
	
	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

	# trans_mat[:] = npy.fliplr(trans_mat[:])
	# trans_mat[:] = npy.flipud(trans_mat[:])
	
	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])

	# trans_mat[1] = trans_mat_1
	# trans_mat[2] = npy.rot90(trans_mat_1)
	# trans_mat[0] = npy.rot90(trans_mat_1,3)
	# trans_mat[3] = npy.rot90(trans_mat_1,2)

	# trans_mat[5] = trans_mat_2
	# trans_mat[7] = npy.rot90(trans_mat_2)
	# trans_mat[6] = npy.rot90(trans_mat_2,2)
	# trans_mat[4] = npy.rot90(trans_mat_2,3)

# conv_transition_filters()

print "Transition Matrices:\n",trans_mat

print "\nHere's the reward.\n"
for i in range(0,discrete_size):
	print reward_function[i]

def action_reward_bias():
	global action_reward_function, action_value_layers

	for act in range(0,action_size):
		action_value_layers[act] += action_reward_function[act]

def conv_layer():	
	global value_function, trans_mat, action_value_layers
	
	layer_value = npy.zeros(shape=(discrete_size,discrete_size))
	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		action_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Fixed bias for reward. 
	action_reward_bias()

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
		# reward_bias()		
		t+=1
		print t
	
recurrent_value_iteration()

print "Here's the policy."
for i in range(0,discrete_size):
	print optimal_policy[i]

# policy_iteration()

# print "These are the value functions."
# for t in range(0,time_limit):
# 	print value_functions[t]
with file('reward_function.txt','w') as outfile: 
	outfile.write('#Reward Function.\n')
	npy.savetxt(outfile,1000*reward_function,fmt='%-7.2f')

with file('output_policy.txt','w') as outfile: 
	outfile.write('#Policy.\n')
	npy.savetxt(outfile,optimal_policy,fmt='%-7.2f')

with file('value_function.txt','w') as outfile: 
	outfile.write('#Value Function.\n')
	npy.savetxt(outfile,1000*value_function,fmt='%-7.2f')

