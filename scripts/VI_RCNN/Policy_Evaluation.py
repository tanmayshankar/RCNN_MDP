#!/usr/bin/env python

from variables import * 

action_factor_reward = 0.2
action_size = 8

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

def action_reward_bias():
	global action_reward_function, action_value_layers

	for act in range(0,action_size):
		q_value_layers[act] += action_reward_function[act]

def conv_layer():	
	global value_function, trans_mat, action_value_layers

	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		q_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Fixed bias for reward. 
	action_reward_bias()

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

# with file('reward_function.txt','w') as outfile: 
# 	outfile.write('#Reward Function.\n')
# 	npy.savetxt(outfile,reward_function,fmt='%-7.2f')

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