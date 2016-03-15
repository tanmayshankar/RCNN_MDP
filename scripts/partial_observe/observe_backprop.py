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
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

#Transition space size determines size of convolutional filters. 
transition_space = 3
obs_space = 3
time_limit = 5000

bucket_space = npy.zeros((action_size,transition_space**2))
cummulative = npy.zeros(action_size)
bucket_index = 0
# time_limit = 500

obs_bucket_space = npy.zeros(obs_space**2)
obs_bucket_index =0 
obs_cummulative = 0

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
corr_to_state_belief = npy.zeros((discrete_size,discrete_size))
#### DEFINING EXTENDED STATE BELIEFS 
w = transition_space/2
to_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))
from_state_ext = npy.zeros((discrete_size+2*w,discrete_size+2*w))

#### DEFINING OBSERVATION RELATED VARIABLES
observation_model = npy.zeros(shape=(obs_space,obs_space))
obs_model_unknown = npy.ones(shape=(obs_space,obs_space))
observed_state = npy.zeros(2)

state_counter = 0
action = 'w'

learning_rate = 0.05
annealing_rate = (learning_rate/5)/time_limit
learning_rate_obs = 0.01
annealing_rate_obs = (learning_rate/5)/time_limit

norm_sum_bel=0.


def initialize_state():
	global current_pose, from_state_belief

	from_state_belief[24,24]=1.
	# from_state_belief[25,24]=0.8
	current_pose=[24,24]


def initialize_transitions():
	global trans_mat
	# trans_mat_1 = npy.array([[0.,0.97,0.],[0.01,0.01,0.01],[0.,0.,0.]])
	# trans_mat_2 = npy.array([[0.97,0.01,0.],[0.01,0.01,0.],[0.,0.,0.]])
	trans_mat_1 = npy.array([[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]])
	trans_mat_2 = npy.array([[0.7,0.1,0.],[0.1,0.1,0.],[0.,0.,0.]])
	
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
	# observation_model = npy.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
	# print observation_model

	epsilon=0.0001
	observation_model += epsilon
	observation_model /= observation_model.sum()

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
	global to_state_belief, current_pose, observation_model, obs_space, observed_state, corr_to_state_belief, norm_sum_bel
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))

	for i in range(-obs_space/2,obs_space/2+1):
		for j in range(-obs_space/2,obs_space/2+1):

			# dummy[observed_state[0]+i,observed_state[1]+j] = to_state_belief[observed_state[0]+i,observed_state[1]+j] * observation_model[obs_space/2+i,obs_space/2+j]
			##MUST INVOKE THE UNKNOWN OBVS
			dummy[observed_state[0]+i,observed_state[1]+j] = to_state_belief[observed_state[0]+i,observed_state[1]+j] * obs_model_unknown[obs_space/2+i,obs_space/2+j]

	corr_to_state_belief[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())
	norm_sum_bel = dummy.sum()

def remap_indices(dummy_index):

	#####action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
	#####UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

	if (dummy_index==0):
		return 4
	if (dummy_index==1):
		return 0
	if (dummy_index==2):
		return 5
	if (dummy_index==3):
		return 2	
	if (dummy_index==5):
		return 3
	if (dummy_index==6):
		return 6
	if (dummy_index==7):
		return 1
	if (dummy_index==8):
		return 7

def initialize_model_bucket():
	global cummulative, bucket_index, bucket_space
	orig_mat = copy.deepcopy(trans_mat)
	for k in range(0,action_size):
		orig_mat = npy.flipud(npy.fliplr(trans_mat[k,:,:]))

		for i in range(0,transition_space):
			for j in range(0,transition_space):
				# Here, it must be the original, non -flipped transition matrix. 
				# cummulative += trans_mat[action_index,transition_space-i,transition_space-j]
				# cummulative += trans_mat[action_index,i,j]
				cummulative[k] += orig_mat[i,j]
				bucket_space[k,transition_space*i+j] = cummulative[k]

def initialize_obs_model_bucket():
	global obs_bucket_space, observation_model, obs_space, obs_cummulative
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			obs_cummulative += observation_model[i,j]
			obs_bucket_space[obs_space*i+j] = obs_cummulative

	print obs_bucket_space

def initialize_all():
	initialize_state()
	initialize_observation()
	initialize_transitions()
	initialize_unknown_observation()
	initialize_unknown_transitions()
	initialize_model_bucket()
	initialize_obs_model_bucket()

def construct_from_ext_state():
	global from_state_ext, from_state_belief,discrete_size
	d=discrete_size
	from_state_ext[w:d+w,w:d+w] = copy.deepcopy(from_state_belief[:,:])
	# from_state_ext[2*w:d+2*w,2*w:d+2*w] = from_state_belief[:,:]

def belief_prop_extended(action_index):
	global trans_mat_unknown, from_state_ext, to_state_ext, w, discrete_size
	to_state_ext = signal.convolve2d(from_state_ext,trans_mat_unknown[action_index],'same')
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

def simulated_model(action_index):
	global trans_mat, from_state_belief, bucket_space, bucket_index, cummulative

	#### BASED ON THE TRANSITION MODEL CORRESPONDING TO ACTION_INDEX, PROBABILISTICALLY FIND THE NEXT SINGLE STATE.
	#must find the right bucket

	rand_num = random.random()

	if (rand_num<bucket_space[action_index,0]):
		bucket_index=0
	
	for i in range(1,transition_space**2):
		if (bucket_space[action_index,i-1]<=rand_num)and(rand_num<bucket_space[action_index,i]):
			bucket_index=i

	remap_index = remap_indices(bucket_index)
	
	if (bucket_index!=((transition_space**2)/2)):
		current_pose[0] += action_space[remap_index][0]
		current_pose[1] += action_space[remap_index][1]
				
	target_belief[:,:] = 0. 
	target_belief[current_pose[0],current_pose[1]]=1.
	 
def simulated_observation_model():
	global observation_model, obs_bucket_space, obs_bucket_index, observed_state, current_pose
	
	remap_index = 0
	rand_num = random.random()
	if (rand_num<obs_bucket_space[0]):
		obs_bucket_index=0
	
	for i in range(1,obs_space**2):
		if (obs_bucket_space[i-1]<=rand_num)and(rand_num<obs_bucket_space[i]):
			obs_bucket_index=i
	
	obs_bucket_index = int(obs_bucket_index)
	observed_state = copy.deepcopy(current_pose)

	if (obs_bucket_index!=((obs_space**2)/2)):
		remap_index = remap_indices(obs_bucket_index)
		observed_state[0] += action_space[remap_index,0]
		observed_state[1] += action_space[remap_index,1]

	# print "Observed State: ", observed_state

def belief_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief	

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()
	# from_state_belief = to_state_belief

def back_prop_trans(action_index,time_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, observation_model, observed_state, corr_to_state_belief

	# loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = learning_rate - annealing_rate * time_index
	# alpha = learning_rate
	lamda = 1.

	w = transition_space/2
	x = copy.deepcopy(observed_state[0])
	y = copy.deepcopy(observed_state[1])

	for m in range(-w,w+1):
		for n in range(-w,w+1):
			loss_1=0.
			for i in range(0,discrete_size):
				for j in range(0,discrete_size):
					# if (((i-m)>=0)and((i-m)<discrete_size)and((j-n)>=0)and((j-n)<discrete_size)and((i-observed_state[0])<obs_space)and((j-observed_state[1])<obs_space)and((j-observed_state[1])>=0)and((i-observed_state[0])>=0)):
					if (i-m>=0)and(i-m<discrete_size)and(j-n>=0)and(j-n<discrete_size):
						if (i-x>=0)and(j-y>=0)and(j-y<obs_space)and(i-x<obs_space):

							loss_1 -= 2*(target_belief[i,j]-corr_to_state_belief[i,j])*observation_model[i-observed_state[0],j-observed_state[1]] * from_state_belief[i-m,j-n] 
			
			# temp = trans_mat_unknown[action_index,w+m,w+n] - alpha*loss[w+m,w+n]		
			# if (temp>=0)and(temp<=1):
			if (trans_mat_unknown[action_index,w+m,w+n] - alpha*loss_1>=0)and(trans_mat_unknown[action_index,w+m,w+n] - alpha*loss_1<1):
				trans_mat_unknown[action_index,w+m,w+n] -= alpha*loss_1

	# trans_mat_unknown[action_index,:,:] /=trans_mat_unknown[action_index,:,:].sum()
def back_prop_obs(action_index,time_index):
	global obs_model_unknown, obs_space, to_state_belief, target_belief, from_state_belief, corr_to_state_belief, norm_sum_bel
	alpha = learning_rate_obs - annealing_rate_obs * time_index

	h = obs_space/2

	for m in range(-h,h+1):
		for n in range(-h,h+1):
			loss_1 =0.

			if (observed_state[0]+m>=0)and(observed_state[0]+m<obs_space)and(observed_state[1]+n<obs_space)and(observed_state[1]+n>=0):
				loss_1 = - 2 * (target_belief[observed_state[0]+m,observed_state[1]+n]-corr_to_state_belief[observed_state[0]+m,observed_state[1]+n]) * to_state_belief[observed_state[0]+m,observed_state[1]+n] / norm_sum_bel

			if (obs_model_unknown[h+m,h+n]-alpha*loss_1>=0)and(obs_model_unknown[h+m,h+n]-alpha*loss_1<1):
				obs_model_unknown[h+m,h+n] -=alpha * loss_1


def recurrence():
	global from_state_belief,target_belief
	from_state_belief = copy.deepcopy(target_belief)

def master(action_index, time_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	# belief_prop(action_index)
	construct_from_ext_state()
	belief_prop_extended(action_index)
	bayes_obs_fusion()

	simulated_model(action_index)
	simulated_observation_model()

	back_prop_trans(action_index, time_index)
	back_prop_obs(action_index, time_index)
	recurrence()	

initialize_all()

def input_actions():
	global action, state_counter, action_index, current_pose

	iterate=0

	while (iterate<=time_limit):		
		iterate+=1
		# action_index = random.randrange(0,8)
		action_index=iterate%8
		print "Iteration:",iterate," Current pose:",current_pose," Observed State:",observed_state," Action:",action_index
		master(action_index, iterate)

input_actions()


def flip_trans_again():
	for i in range(0,action_size):
		trans_mat_unknown[i] = npy.fliplr(trans_mat_unknown[i])
		trans_mat_unknown[i] = npy.flipud(trans_mat_unknown[i])

flip_trans_again()

print "Transition Matrix: "
print trans_mat_unknown

# for i in range(0,8):
# 	print trans_mat_unknown[i].sum()

	
# trans_mat_unknown[:,:,:] /=trans_mat_unknown[:,:,:].sum()
# print "Normalized:\n",trans_mat_unknown	
print "Actual transition matrix:" , trans_mat

print "Learnt Observation Model:", obs_model_unknown























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



with file('actual_transition.txt','w') as outfile: 
	for data_slice in trans_mat:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')

with file('estimated_transition.txt','w') as outfile: 
	for data_slice in trans_mat_unknown:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')

# with file('output_policy.txt','w') as outfile: 
# 	outfile.write('#Policy.\n')
# 	npy.savetxt(outfile,optimal_policy,fmt='%-7.2f')

# with file('value_function.txt','w') as outfile: 
# 	outfile.write('#Value Function.\n')
# 	npy.savetxt(outfile,1000*value_function,fmt='%-7.2f')
