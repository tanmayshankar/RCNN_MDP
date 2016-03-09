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


###### DEFINITIONS

basis_size = 3
discrete_size = 50

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

#Transition space size determines size of convolutional filters. 
transition_space = 3
time_limit = 1000

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

#### DEFINING OBSERVATION RELATED VARIABLES
obs_space = 3
observation_model = npy.zeros(shape=(obs_space,obs_space))
obs_model_unknown = npy.ones(shape=(obs_space,obs_space))

state_counter = 0
action = 'w'

learning_rate = 0.05
annealing_rate = (learning_rate/5)/time_limit


def initialize_state():
	global current_pose, from_state_belief

	from_state_belief[24,24]=1.
	# from_state_belief[25,24]=0.8
	current_pose=[24,24]


def initialize_transitions():
	global trans_mat
	trans_mat_1 = npy.array([[0.,0.97,0.],[0.01,0.01,0.01],[0.,0.,0.]])
	trans_mat_2 = npy.array([[0.97,0.01,0.],[0.01,0.01,0.],[0.,0.,0.]])
	
	#Adding epsilon so that the cummulative distribution has unique values. 
	epsilon=0.001
	trans_mat_1+=epsilon
	trans_mat_2+=epsilon

	trans_mat_1/=trans_mat_1.sum()
	trans_mat_2/=trans_mat_2.sum()

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

	print "Transition Matrices:\n",trans_mat

	# for i in range(0,action_size):
	# 	trans_mat[i] = npy.fliplr(trans_mat[i])
	# 	trans_mat[i] = npy.flipud(trans_mat[i])

	

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
	# global from_state_belief
	# global current_pose
	# global observation_model
	
	global from_state_belief, current_pose, observation_model

	dummy = npy.zeros(shape=(discrete_size,discrete_size))

	# l = obs_space/2
	# for i in range(-l,l+1):
	# 	for j in range(-l,l+1):
	# 		dummy[i+current_pose[0],j+current_pose[1]] = from_state_belief[i+current_pose[0],j+current_pose[1]]*observation_model[l+i,l+j]

	for i in range(0,obs_space):
		for j in range(0,obs_space):
			dummy[current_pose[0]-1+i,current_pose[1]-1+j] = from_state_belief[current_pose[0]-1+i,current_pose[1]-1+j]*observation_model[i,j]

	# print "Dummy.",dummy
	from_state_belief[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())

def display_beliefs():
	global from_state_belief,to_state_belief,target_belief,current_pose

	# print "From:"
	# for i in range(20,30):
	# 	print from_state_belief[50-i,20:30]
	# # for i in range(1,50):
	# # 	print from_state_belief[50-i,:]
	# print "To:"
	# for i in range(20,30):
	# 	print to_state_belief[50-i,20:30]
	# # for i in range(1,50):
	# # 	print to_state_belief[50-i,:]
	# print "Target:",
	# for i in range(20,30):
	# 	print target_belief[50-i,20:30]
	# # for i in range(1,50):	
	# # 	print target_belief[50-i,:]

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
	global to_state_belief
	global current_pose
	global observation_model
	global obs_space
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))

	for i in range(0,obs_space):
		for j in range(0,obs_space):
			dummy[current_pose[0]-1+i,current_pose[1]-1+j] = to_state_belief[current_pose[0]-1+i,current_pose[1]-1+j]*observation_model[i,j]
	
	to_state_belief[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())

def calculate_target(action_index):
	# global trans_mat_unknown
	# global to_state_belief
	# global from_state_belief
	global target_belief

	#TARGET TYPE 1 
	# target_belief[:,:]=0.
	# target_belief[to_state[0],to_state[1]]=1.

	#TARGET TYPE 2: actual_T * from_belief
	# target_belief = from_state_belief
	target_belief = signal.convolve2d(from_state_belief,trans_mat[action_index],'same','fill',0)
	
	#TARGET TYPE 3: 
	# target_belief = from_state_belief
	# target_belief = signal.convolve2d(from_state_belief,trans_mat[action_index],'same','fill',0)
	#Fuse with Observations

	#TARGET TYPE 4: 
	

	# if (target_belief.sum()<1.):
	# 	target_belief /= target_belief.sum()

def remap_indices(bucket_index):

	#####action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
	#####UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

	if (bucket_index==0):
		return 4
	elif (bucket_index==1):
		return 0
	elif (bucket_index==2):
		return 5
	elif (bucket_index==3):
		return 2	
	elif (bucket_index==5):
		return 3
	elif (bucket_index==6):
		return 6
	elif (bucket_index==7):
		return 1
	elif (bucket_index==8):
		return 7

def simulated_model(action_index):
	global trans_mat, from_state_belief

	#### BASED ON THE TRANSITION MODEL CORRESPONDING TO ACTION_INDEX, PROBABILISTICALLY FIND THE NEXT SINGLE STATE.
	#must find the right bucket

	rand_num = random.random()
	bucket_space = npy.zeros(transition_space**2)
	cummulative = 0.
	bucket_index =0

	orig_mat = npy.flipud(npy.fliplr(trans_mat[action_index,:,:]))

	for i in range(0,transition_space):
		for j in range(0,transition_space):
			# Here, it must be the original, non -flipped transition matrix. 
			# cummulative += trans_mat[action_index,transition_space-i,transition_space-j]
			# cummulative += trans_mat[action_index,i,j]
			cummulative += orig_mat[i,j]
			bucket_space[transition_space*i+j] = cummulative


	if (rand_num<bucket_space[0]):
		bucket_index=0
	# elif (rand_num>bucket_space[7]):
		# bucket_index=8
	# else:

	# print "BUCKET SPACE:",bucket_space
	# print "Random:",rand_num
	
	for i in range(1,transition_space**2):
		if (bucket_space[i-1]<=rand_num)and(rand_num<bucket_space[i]):
			bucket_index=i
			# print "Bucket Index chosen: ",bucket_index

	remap_index = remap_indices(bucket_index)
	# print "Remap Index:",remap_index
	# print "Action Index: ",action_index," Ideal Action: ",action_space[action_index]

	# if (bucket_index==((transition_space**2)/2)):
		# print "Bucket index: ",bucket_index, "Action taken: ","[0,0]"
		# print "No action."		
	# else:
	if (bucket_index!=((transition_space**2)/2)):
		current_pose[0] += action_space[remap_index][0]
		current_pose[1] += action_space[remap_index][1]
		
		# print "Remap index: ",remap_index, "Action taken: ",action_space[remap_index]		
		
	target_belief[:,:] = 0. 
	target_belief[current_pose[0],current_pose[1]]=1.
	
	

def belief_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief	

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()
	# from_state_belief = to_state_belief

dummy_from_belief = npy.zeros((discrete_size+2,discrete_size+2))
def calc_dummy_from_belief():
	global from_state_belief, dummy_from_belief
	for i in range(0,discrete_size):
		for j in range(0,discrete_size):
			dummy_from_belief[1+i,1+j]=from_state_belief[i,j]

def back_prop(action_index, time_index):
	# global trans_mat_unknown
	# global to_state_belief
	# global from_state_belief
	# global target_belief

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, dummy_from_belief

	loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.01

	alpha = learning_rate - annealing_rate * time_index

	lamda = 1.

	w = transition_space/2
	# print w
	delta = 0.
	for ai in range(-w,w+1):
		for aj in range(-w,w+1):
			
			# loss[w+ai,w+aj] += lamda * (trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
					
			for i in range(0,discrete_size):
				for j in range(0,discrete_size):
					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj])
					# delta = (trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j]) 

					#*(from_state_belief[w+i-ai,w+j-aj]) #+ delta
					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[i,j]) #+ delta
					temp_1 = 0.
					if (w+i-ai>=50)or(w+i-ai<0)or(w+j-aj>=50)or(w+j-aj<0):
						temp_1 =0.
					else:
						temp_1 = from_state_belief[w+i-ai,w+j-aj]
					loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*temp_1
					
					
			# trans_mat_unknown[action_index,w+ai,w+aj] += alpha * loss[w+ai,w+aj]
			temp = trans_mat_unknown[action_index,w+ai,w+aj] - alpha * loss[w+ai,w+aj]
			if (temp<=1)and(temp>=0):
				# trans_mat_unknown[action_index,w+m,w+n]=temp
				trans_mat_unknown[action_index,w+ai,w+aj] = temp
			# trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
			# if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
			# 	trans_mat_unknown[action_index,w+ai,w+aj]=0
			# trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()
	trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()

def recurrence():
	global from_state_belief,target_belief
	from_state_belief = copy.deepcopy(target_belief)

def master(action_index, time_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	# belief_prop(action_index)
	# # bayes_obs_fusion()
	# simulated_model(action_index)
	# back_prop(action_index)
	# recurrence()	


	###Fiddling with the order: 

	
	# bayes_obs_fusion()
	# display_beliefs()
	simulated_model(action_index)	
	belief_prop(action_index)	
	calc_dummy_from_belief()
	back_prop(action_index, time_index)
	recurrence()	




	
	# print "current_pose:",current_pose
	print "Transition Matrix: ",action_index,"\n"
	print trans_mat_unknown[action_index,:,:]

	# print npy.flipud(npy.fliplr(trans_mat_unknown[action_index,:,:]))

initialize_all()

def input_actions():
	global action, state_counter, action_index, current_pose

	# while (action!='q'):		
	iterate=0

	while (iterate<=time_limit):		
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


		master(action_index, iterate)

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
