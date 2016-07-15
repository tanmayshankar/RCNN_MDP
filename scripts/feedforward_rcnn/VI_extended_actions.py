#!/usr/bin/env python

from variables import *

reward_function = reward_function.reshape((action_size,discrete_size,discrete_size))
action_reward_function = copy.deepcopy(reward_function)
action_size = 9 

time_limit = 40

trans_mat_1 = npy.zeros((action_size,transition_space,transition_space))

for i in range(0,action_size-1):
	trans_mat_1[i] = trans_mat[i]
trans_mat_1[action_size-1,1,1]=1.
trans_mat = copy.deepcopy(trans_mat_1)

def modify_trans_mat(): 
	global trans_mat
	epsilon = 0.0001
	for i in range(0,action_size):
		trans_mat[i][:][:] += epsilon
		trans_mat[i] /= trans_mat[i].sum()

	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])

def initialize():
	modify_trans_mat()
	create_action_reward()

initialize()

def action_reward_bias():
	global action_reward_function, action_value_layers, q_value_layers

	for act in range(0,action_size):		
		q_value_layers[act] += action_reward_function[act]

def conv_layer():	
	global value_function, trans_mat, action_value_layers, q_value_layers

	for act in range(0,action_size):		
		#Convolve with each transition matrix.		
		q_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Fixed bias for reward. 
	action_reward_bias()

	#Max pooling over actions. 
	value_function = gamma*npy.amax(q_value_layers,axis=0)
	optimal_policy[:,:] = npy.argmax(q_value_layers,axis=0)

def recurrent_value_iteration():
	global value_function
	t=0	
	while (t<time_limit):
		conv_layer()
		t+=1
		print t
	
recurrent_value_iteration()

def bound_policy():
	## FOR ORIGINAL:
	optimal_policy[0,:] = 1
	optimal_policy[49,:] = 0
	optimal_policy[:,0] = 3
	optimal_policy[:,49] = 2
	optimal_policy[0,0] = 7
	optimal_policy[0,49] = 6
	optimal_policy[49,0] = 5
	optimal_policy[49,49] = 4

bound_policy()

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

with file('Q_Value_Function.txt','w') as outfile: 
	for data in q_value_layers:
		outfile.write('#Q Value Function.\n')
		npy.savetxt(outfile,data,fmt='%-7.2f')