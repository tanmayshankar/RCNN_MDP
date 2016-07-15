#!/usr/bin/env python

from variables import * 

# import cProfile
# cProfile.run('foo()')

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
		
modify_trans_mat()

def conv_layer():	
	global value_function, trans_mat

	action_value_layers = npy.zeros(shape=(action_size,discrete_size,discrete_size))
	layer_value = npy.zeros(shape=(discrete_size,discrete_size))
	
	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		action_value_layers[act]=signal.convolve2d(value_function,trans_mat[act],'same','fill',0)
	
	#Max pooling over actions. 
	value_function = gamma*npy.amax(action_value_layers,axis=0)
	optimal_policy[:,:] = npy.argmax(action_value_layers,axis=0)

def reward_bias():
	global value_function
	value_function += reward_function

def recurrent_value_iteration():
	global value_function
	print "Start iterations."
	t=0	
	while (t<time_limit):
		conv_layer()
		reward_bias()		
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

print "The policy is as follows:"
for i in range(0,discrete_size):
	print optimal_policy[i]

with file('reward_function.txt','w') as outfile: 
	outfile.write('#Reward Function.\n')
	npy.savetxt(outfile,1000*reward_function,fmt='%-7.2f')

with file('output_policy.txt','w') as outfile: 
	outfile.write('#Policy.\n')
	npy.savetxt(outfile,optimal_policy,fmt='%-7.2f')

with file('value_function.txt','w') as outfile: 
	outfile.write('#Value Function.\n')
	npy.savetxt(outfile,1000*value_function,fmt='%-7.2f')

