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

print "Started."
###### DEFINITIONS

basis_size = 3
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

#Transition space size determines size of convolutional filters. 
transition_space = 3
time_limit = 100

# npy.set_printoptions(precision=3)

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

#### DEFINING OBSERVATION RELATED VARIABLES
obs_space = 3
observation_model = npy.zeros(shape=(obs_space,obs_space))
obs_model_unknown = npy.ones(shape=(obs_space,obs_space))

state_counter = 0
action = 'w'
norm_factor=0.


def initialize_state():
	global current_pose, from_state_belief

	from_state_belief[24,24]=1.
	# from_state_belief[25,24]=0.8
	current_pose=[24,24]

def initialize_transitions():
	global trans_mat
	trans_mat_1 = [[0.,0.97,0.],[0.01,0.01,0.01],[0.,0.,0.]]
	trans_mat_2 = [[0.97,0.01,0.],[0.01,0.01,0.],[0.,0.,0.]]

	# trans_mat_1 = [[0.,0.,0.1,0.,0.],[0.,0.05,0.6,0.05,0.],[0.,0.05,0.1,0.05,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
	# trans_mat_2 = [[0.1,0.05,0.,0.,0.],[0.05,0.6,0.05,0.,0.],[0.,0.05,0.1,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]
	# trans_mat_1 = [[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]]
	# trans_mat_2 = [[0.7,0.1,0.],[0.1,0.1,0.],[0.,0.,0.]]
	
	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])

	print "Transition Matrices:\n",trans_mat

def initialize_unknown_transitions():
	global trans_mat_unknown

	for i in range(0,transition_space):
		for j in range(0,transition_space):
	# 		trans_mat_unknown[:,i,j] = random.random()
			trans_mat_unknown[:,i,j] = 1.
	for i in range(0,action_size):
		trans_mat_unknown[i,:,:] /=trans_mat_unknown[i,:,:].sum()

def initialize_unknown_observation():
	global obs_model_unknown
	obs_model_unknown = obs_model_unknown/obs_model_unknown.sum()

def initialize_observation():
	global observation_model
	observation_model = npy.array([[0.,0.1,0.],[0.1,0.6,0.1],[0.,0.1,0.]])
	print observation_model

def initialize_all():
	initialize_state()
	initialize_observation()
	initialize_transitions()
	initialize_unknown_observation()
	initialize_unknown_transitions()

def fuse_observations():
	global from_state_belief, current_pose, observation_model

	dummy = npy.zeros(shape=(discrete_size,discrete_size))

	# l = obs_space/2
	# for i in range(-l,l+1):
	# 	for j in range(-l,l+1):
	# 		dummy[i+current_pose[0],j+current_pose[1]] = from_state_belief[i+current_pose[0],j+current_pose[1]]*observation_model[l+i,l+j]
	h = obs_space/2
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			dummy[current_pose[0]-1+i,current_pose[1]-1+j] = from_state_belief[current_pose[0]-1+i,current_pose[1]-1+j]*observation_model[i,j]

	# print "Dummy.",dummy
	from_state_belief[:,:] = dummy[:,:]/dummy.sum()

def bayes_obs_fusion():
	global to_state_belief, current_pose, observation_model, obs_space, norm_factor
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))
	h = obs_space/2
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			dummy[current_pose[0]-h+i,current_pose[1]-h+j] = to_state_belief[current_pose[0]-h+i,current_pose[1]-h+j]*observation_model[i,j]

	# for i in range(-l,l+1):
	# 	for j in range(-l,l+1):
	# 		dummy[i+current_pose[0],j+current_pose[1]] = from_state_belief[i+current_pose[0],j+current_pose[1]]*observation_model[l+i,l+j]

	norm_factor = dummy.sum()				
	to_state_belief[:,:] = dummy[:,:]/dummy.sum()

def bayes_fusion_target():
	global target_belief, current_pose, observation_model, obs_space
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))
	h = obs_space/2
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			dummy[current_pose[0]-h+i,current_pose[1]-h+j] = target_belief[current_pose[0]-h+i,current_pose[1]-h+j]*observation_model[i,j]
	
	target_belief[:,:] = dummy[:,:]/dummy.sum()

def calculate_target(action_index):
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
	# 	target_belief /= target_belief.sum()

def simulated_model(action_index):
	global trans_mat, from_state_belief

	#### BASED ON THE TRANSITION MODEL CORRESPONDING TO ACTION_INDEX, PROBABILISTICALLY FIND THE NEXT SINGLE STATE.
	#must find the right bucket

	rand_num = random.random()
	bucket_space = npy.zeros(transition_space**2)
	cummulative = 0.
	bucket_index =0

	for i in range(0,transition_space):
		for j in range(0,transition_space):
			cummulative += trans_mat[action_index,i,j]
			bucket_space[transition_space*i+j] = cummulative

	if (rand_num<bucket_space[0]):
		bucket_index=0
	elif (rand_num>bucket_space[8]):
		bucket_index=8
	else:
		for i in range(1,transition_space**2):
			if (bucket_space[i-1]<rand_num)and(rand_num<bucket_space[i]):
				bucket_index=i

	if (bucket_index<(transition_space/2)):
		# target_belief[:,:]=0.
		current_pose[0] += action_space[bucket_index][0]
		current_pose[1] += action_space[bucket_index][1]
		# target_belief[current_pose[0],current_pose[1]]=1.

	elif (bucket_index>(transition_space/2)):
		# target_belief[:,:]=0.
		current_pose[0] += action_space[bucket_index-1][0]
		current_pose[1] += action_space[bucket_index-1][1]
		# target_belief[current_pose[0],current_pose[1]]=1.
	
	target_belief[:,:] = 0. 
	target_belief[current_pose[0],current_pose[1]]=1.

def belief_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief	

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()
	# from_state_belief = to_state_belief

def display_beliefs():
	global from_state_belief,to_state_belief,target_belief

	print "From:"
	for i in range(20,30):
		print from_state_belief[50-i,20:30]
	# for i in range(1,50):
	# 	print from_state_belief[50-i,:]
	print "To:"
	for i in range(20,30):
		print to_state_belief[50-i,20:30]
	# for i in range(1,50):
	# 	print to_state_belief[50-i,:]
	print "Target:",
	for i in range(20,30):
		print target_belief[50-i,20:30]
	# for i in range(1,50):	
	# 	print target_belief[50-i,:]


def back_prop(action_index):
	# global trans_mat_unknown
	# global to_state_belief
	# global from_state_belief
	# global target_belief

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief	

	loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.01
	lamda = 1.
	w = transition_space/2
	display_beliefs()
	difference_term = 0.

	for i in range(0,discrete_size):
		for j in range(0,discrete_size):
			difference_term -= (target_belief[i,j]-to_state_belief[i,j])

	difference_term*=2

	delta = 0.
	for ai in range(-w,w+1):
		for aj in range(-w,w+1):
			
			loss[w+ai,w+aj] += lamda *(trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
			
			for i in range(0,discrete_size-2):
				for j in range(0,discrete_size-2):

					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj])
					# delta = (trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
					loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj]) #+ delta
					

			# trans_mat_unknown[action_index,w+ai,w+aj] += alpha * loss[w+ai,w+aj]
			trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
			# if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
			# 	trans_mat_unknown[action_index,w+ai,w+aj]=0
			# trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()
	trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()

def trans_back_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief	
	global transition_space,obs_space, norm_factor, current_pose

	trans_loss_grad = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.1
	
	w = transition_space/2		
	h = obs_space/2
	#Defining weightage for the sum of transition.
	lamda = 1.

	display_beliefs()

	difference_term = 0.
	obs_mask_term =0.
	# obs_mask_term = npy.zeros(shape=(obs_space,obs_space))
	sum_term = 0.

	for i in range(0,discrete_size):
		for j in range(0,discrete_size):
			difference_term -= (target_belief[i,j]-to_state_belief[i,j])

	difference_term *= 2/norm_factor

	sum_term = lamda*((trans_mat_unknown[action_index,:,:].sum()-1)**2)

	for ai in range(-w,w+1):
		for aj in range(-w,w+1):
			
			# loss[w+ai,w+aj]+=lamda*(trans_mat_unknown[action_index,:,:].sum()-1)**2
			# for i in range(0,discrete_size):
			# 	for j in range(0,discrete_size):
			# 		term -= 2*(target_belief[i,j]-to_state_belief[i,j])*norm_factor

			for k in range(-h,h+1):
				for l in range(-h,h+1):
					# obs_mask_term[k+h,l+h] += observation_model[h+k,h+l] * from_state_belief[current_pose[0]+k+w-ai,current_pose[1]+l+w-aj]
					obs_mask_term += observation_model[h+k,h+l] * from_state_belief[current_pose[0]+k+w-ai,current_pose[1]+l+w-aj]
							# trans_loss_grad[w+ai,w+aj] -= 

			trans_loss_grad[w+ai,w+aj] = difference_term*obs_mask_term + sum_term*trans_mat_unknown[action_index,w+ai,w+aj] 
			trans_mat_unknown[action_index,w+ai,w+aj] += alpha * trans_loss_grad[w+ai,w+aj]

			# trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
			# if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
			# 	trans_mat_unknown[action_index,w+ai,w+aj]=0

	# trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()

def recurrence():
	global from_state_belief,target_belief
	from_state_belief = target_belief

def master(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	belief_prop(action_index)
	# bayes_obs_fusion()
	simulated_model(action_index)
	# bayes_fusion_target()
	# trans_back_prop(action_index)
	back_prop(action_index)
	recurrence()	
	
	# print "current_pose:",current_pose
	# print "Transition Matrix: ",action_index,"\n"
	# print trans_mat_unknown[action_index,:,:]
	# print npy.flipud(npy.fliplr(trans_mat_unknown[action_index,:,:]))

initialize_all()

def input_actions():
	global action
	global state_counter
	global action_index
	global current_pose

	# while (action!='q'):		
	iterate=0
	# while (iterate<=5000):		
	# ############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........
	# 		# 
	# 		iterate+=1
	# 		# action = raw_input("Hit a key now: ")
	# 		# action = 
	# 		if action=='w':			
	# 			state_counter+=1	
	# 			# current_demo.append([current_pose[0]+1,current_pose[1]])
	# 			current_pose[0]+=1			
	# 			action_index=0

	# 		if action=='a':			
	# 			state_counter+=1		
	# 			# current_demo.append([current_pose[0],current_pose[1]-1])
	# 			current_pose[1]-=1			
	# 			action_index=2

	# 		if action=='d':			
	# 			state_counter+=1
	# 			# current_demo.append([current_pose[0],current_pose[1]+1])
	# 			current_pose[1]+=1
	# 			action_index=1

	# 		if action=='s':			
	# 			state_counter+=1
	# 			# current_demo.append([current_pose[0]-1,current_pose[1]])
	# 			current_pose[0]-=1
	# 			action_index=3

	# 		if ((action=='wa')or(action=='aw')):			
	# 			state_counter+=1	
	# 			# current_demo.append([current_pose[0]+1,current_pose[1]-1])
	# 			current_pose[0]+=1	
	# 			current_pose[1]-=1						
	# 			action_index=4

	# 		if ((action=='sa')or(action=='as')):					
	# 			state_counter+=1		
	# 			# current_demo.append([current_pose[0]-1,current_pose[1]-1])
	# 			current_pose[1]-=1
	# 			current_pose[0]-=1		
	# 			action_index=6

	# 		if ((action=='sd')or(action=='ds')):					
	# 			state_counter+=1
	# 			# current_demo.append([current_pose[0]-1,current_pose[1]+1])
	# 			current_pose[1]+=1
	# 			current_pose[0]-=1
	# 			action_index=7

	# 		if ((action=='wd')or(action=='dw')):					
	# 			state_counter+=1
	# 			# current_demo.append([current_pose[0]+1,current_pose[1]+1])
	# 			current_pose[0]+=1			
	# 			current_pose[1]+=1			
	# 			action_index=5

	# 		# path_plot[current_pose[0]][current_pose[1]]=1				
	# 		master(action_index)

	while (iterate<=100):		
		iterate+=1
		# select_action()
		# print iterate

		# action_index = random.randrange(0,8)
		action_index=iterate%8
		# dum_x = current_pose[0] + action_space[action_index][0]
		# dum_y = current_pose[1] + action_space[action_index][1]

		# # if ((dum_x<50)and(dum_x>=0)and(dum_y<50)and(dum_y>=0)):
		# if ((dum_x<49)and(dum_x>=1)and(dum_y<49)and(dum_y>=1)):
		# 	current_pose[0]=dum_x
		# 	current_pose[1]=dum_y

		print "Iteration:",iterate," Current pose:",current_pose," Action:",action_index


		master(action_index)

input_actions()

print trans_mat_unknown















######TO RUN FEEDFORWARD PASSES OF THE RECURRENT CONV NET.#########

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






######TO SAVE THE POLICY AND VALUE FUNCTION:######

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