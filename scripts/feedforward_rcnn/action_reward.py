#!/usr/bin/env python
import numpy as npy
# import matplotlib.pyplot as plt
# import rospy
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

##THE ORIGINAL ACTION SPACE:
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
##THE MODIFIED ACTION SPACE:
# action_space = npy.array([[1,0],[-1,0],[0,-1],[0,1],[1,-1],[1,1],[-1,-1],[-1,1]])
################# UP,  DOWN,  LEFT, RIGHT,UPLEFT,UPRIGHT,DOWNLEFT,DOWNRIGHT ##################

#Transition space size determines size of convolutional filters. 
transition_space = 3

#Static / instantaneous reward. 
reward_function = npy.loadtxt(str(sys.argv[1]))

time_limit = 100

dummy_rew = copy.deepcopy(reward_function)
dummy_rew = abs(dummy_rew)
reward_function /= npy.amax(dummy_rew)

action_factor_reward = 0.2

value_functions = npy.zeros(shape=(time_limit,discrete_size,discrete_size))
value_function = npy.zeros(shape=(discrete_size,discrete_size))
optimal_policy = npy.zeros(shape=(discrete_size,discrete_size))

gamma = 0.95
gamma = 0.98
# gamma = 1.

# trans_mat = npy.zeros(shape=(action_size,transition_space,transition_space))
trans_mat = npy.loadtxt(str(sys.argv[2]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

action_reward_function = npy.zeros((action_size,discrete_size,discrete_size))
action_value_layers  = npy.zeros(shape=(action_size,discrete_size,discrete_size))
q_value_layers  = npy.zeros(shape=(action_size,discrete_size,discrete_size))

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

	# for i in range(0,action_size):
	# 	action_reward_function[i,:,:] = copy.deepcopy(reward_function)
		
	for i in range(0,action_size/2):
		action_reward_function[i,:,:] = copy.deepcopy(reward_function) - (i%4)*action_factor_reward * npy.amax(reward_function)
		# print (i%4)

	for i in range(action_size/2,action_size):
		action_reward_function[i,:,:] = copy.deepcopy(reward_function) - 1.414*(i%4)*action_factor_reward * npy.amax(reward_function)

def initialize():
	modify_trans_mat()
	create_action_reward()

initialize()

print "Transition Matrices:\n",trans_mat

print "\nHere's the reward.\n"
for i in range(0,discrete_size):
	print reward_function[i]

def action_reward_bias():
	global action_reward_function, action_value_layers

	for act in range(0,action_size):
		# action_value_layers[act] += action_reward_function[act]
		q_value_layers[act] += action_reward_function[act]

def conv_layer():	
	global value_function, trans_mat, action_value_layers

	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		# action_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
		q_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Fixed bias for reward. 
	action_reward_bias()

	#Max pooling over actions. 
	# value_function = gamma*npy.amax(action_value_layers,axis=0)
	# optimal_policy[:,:] = npy.argmax(action_value_layers,axis=0)

	value_function = gamma*npy.amax(q_value_layers,axis=0)
	optimal_policy[:,:] = npy.argmax(q_value_layers,axis=0)

	print "The next value function.",value_function
	
def reward_bias():
	global value_function
	value_function = value_function + reward_function

def recurrent_value_iteration():
	global value_function
	print "Start iterations."
	t=0	
	while (t<time_limit):
		conv_layer()
		t+=1
		print t
	
recurrent_value_iteration()

print "Here's the policy."
for i in range(0,discrete_size):
	print optimal_policy[i]

##THE ORIGINAL ACTION SPACE:
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
##THE MODIFIED ACTION SPACE:
# action_space = [[1,0],[-1,0],[0,-1],[0,1],[1,-1],[1,1],[-1,-1],[-1,1]]
################# UP,  DOWN,  LEFT, RIGHT,UPLEFT,UPRIGHT,DOWNLEFT,DOWNRIGHT ##################

## FOR ORIGINAL:
optimal_policy[0,:] = 1
optimal_policy[49,:] = 0
optimal_policy[:,0] = 3
optimal_policy[:,49] = 2
optimal_policy[0,0] = 7
optimal_policy[0,49] = 6
optimal_policy[49,0] = 5
optimal_policy[49,49] = 4

# # FOR MODIFIED:
# optimal_policy[0,:] = 0
# # optimal_policy[10,:] = 7
# optimal_policy[49,:] = 1
# optimal_policy[:,0] = 3
# optimal_policy[:,49] = 2
# optimal_policy[0,0] = 5
# optimal_policy[0,49] = 4
# optimal_policy[49,0] = 7
# optimal_policy[49,49] = 6

print "Here's the policy."
for i in range(0,discrete_size):
	print optimal_policy[i]


dummy_policy = copy.deepcopy(optimal_policy)
dummy_policy[1:49,1:49] = 7

with file('reward_function.txt','w') as outfile: 
	outfile.write('#Reward Function.\n')
	npy.savetxt(outfile,reward_function,fmt='%-7.2f')

with file('action_reward_function.txt','w') as outfile: 
	for data in action_reward_function:
		outfile.write('#Action Reward Function.\n')
		npy.savetxt(outfile,data,fmt='%-7.2f')

with file('output_policy.txt','w') as outfile: 
	outfile.write('#Policy.\n')
	npy.savetxt(outfile,optimal_policy,fmt='%-7.2f')

with file('value_function.txt','w') as outfile: 
	outfile.write('#Value Function.\n')
	npy.savetxt(outfile,value_function,fmt='%-7.2f')

with file('Q_value_function.txt','w') as outfile: 
	for data in q_value_layers:
		outfile.write('#Q Value Function.\n')
		npy.savetxt(outfile,data,fmt='%-7.2f')