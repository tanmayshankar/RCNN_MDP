#!/usr/bin/env python

from variables import * 

# import cProfile
# cProfile.run('foo()')

action_size = 8

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
	# value_function += rew_rate*reward_function

def show_image(image_arg):
	plt.imshow(image_arg, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
	plt.show(block=False)
	plt.colorbar()
	plt.show() 

def recurrent_value_iteration():
	global value_function, optimal_policy
	
	t=0	
	X, Y = np.mgrid[0:discrete_size, 0:discrete_size]
	U = npy.zeros(shape=(discrete_size,discrete_size))
	V = npy.zeros(shape=(discrete_size,discrete_size))

	while (t<time_limit):
		conv_layer()
		reward_bias()		
		
		t+=1

		if (t%10==0):
			
			optimal_policy=optimal_policy.astype(int)
			
			for i in range(0,discrete_size):
				for j in range(0,discrete_size):
					U[i,j] = action_space[optimal_policy[i,j]][0]
					V[i,j] = action_space[optimal_policy[i,j]][1]		

			fig, ax = plt.subplots()			
			im = ax.imshow(value_function, origin='lower',extent=[0,49,0,49])
			ax.quiver(V,U)
			fig.colorbar(im)			
			plt.show()	
	
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

