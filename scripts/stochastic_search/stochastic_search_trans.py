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
import heapq

var_size = 9

decision_variables = npy.zeros(var_size)

min_value = 0.
max_value =  1. 

pop_size = 20
population = npy.zeros(shape=(pop_size,var_size))

# function_values = npy.zeros(shape=(4,pop_size))
function_values = npy.zeros(4*pop_size)
master_function_values = npy.zeros(4*pop_size)
mod_func_values = npy.zeros(pop_size)
rel_weights = npy.zeros(pop_size)

sigma = 0.2

neighbour_children = npy.zeros(shape=(pop_size,var_size))
num_pop_children = npy.zeros(pop_size)
num_pop_children = num_pop_children.astype(int)
lc_children = npy.zeros(shape=(pop_size,var_size))
rri_children = npy.zeros(shape=(pop_size,var_size))



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

# previous_pose = copy.deepcopy(current_pose)

def calc_mod_values():
	global mod_func_values, function_values
	# mod_func_values = function_values[0,:] - npy.amin(function_values[0,:])
	mod_func_values = function_values[0:pop_size] - npy.amin(function_values[0:pop_size])
	print "Mod values:",mod_func_values

def calc_rel_weights():
	global rel_weights,mod_func_values
	# rel_weights = mod_func_values / mod_func_values.sum()
	rel_weights[:] = mod_func_values[:] / mod_func_values.sum()
	print "Weight values:",rel_weights

def calc_neighbour_children():
	global neighbour_children, population, num_pop_children
	
	# for i in range(0,pop_size):
	# 	rand_num = random.random()
	# 	pop_children[i,:] = population[i,:] + sigma*rand_num

	# for i in range(0,pop_size):
	# 	for j in range(0,var_size):
	# 		rand_num = random.random()
	# 		pop_children[i,j] = population[i,j] + rand_num*sigma
	# print num_pop_children
	num_pop_children = rel_weights * pop_size
	# print "Number of children:", num_pop_children
	# print "Sum:", num_pop_children.sum()
	# print num_pop_children
	for i in range(0,pop_size):
		num_pop_children[i] = round(num_pop_children[i])

	# print "Number of children:", num_pop_children
	# print "Sum:", num_pop_children.sum()

	i=0
	while (num_pop_children.sum()>20):
		if (num_pop_children[i]>0):
			num_pop_children[i]-=1
		i+=1
	i=0
	while (num_pop_children.sum()<20):
		if (num_pop_children[i]==0):
			num_pop_children[i]+=1
		i+=1

	num_pop_children = num_pop_children.astype(int)	
	child_counter = 0.

	# for i in range(0,pop_size):
	# 	for j in range(0,num_pop_children[i]):
	# 		for k in range(0,var_size):
	# 			rand_num = random.random()
	# 			neighbour_childre
	# while (num_pop_children.sum()>20):
	# 	if (num_pop_children[i]>0):
	# 		num_pop_children[i]-=1
	# 	i+=1n[child_counter,k] = rand_num * sigma + population[i,k]
	# 		child_counter+=1

	for i in range(0,pop_size):
		for j in range(0,num_pop_children[i]):
			rand_num = random.random()
			for k in range(0,var_size):
				neighbour_children[child_counter,k] = rand_num * sigma + population[i,k]
			child_counter+=1

def calc_lc_children():
	global population, lc_children, pop_size

	for i in range(0,pop_size):
		rand_wt = random.random()
		rand_ind_1 = random.randrange(0,pop_size)
		rand_ind_2 = random.randrange(0,pop_size)
		lc_children[i,:] = rand_wt*population[rand_ind_1,:] + (1-rand_wt)*population[rand_ind_2,:]

def calc_rri_children():
	global population,rri_children, pop_size

	for i in range(0,pop_size):
		for j in range(0,var_size):
			rand_num=random.random()
			rri_children[i,j]= min_value + rand_num * (max_value-min_value)

def initialize_pop():
	global population,rri_children,pop_size
	# calc_rri_children()
	# population=rri_children
	for i in range(0,transition_space):
		for j in range(0,transition_space):
			population[3*i+j] = trans_mat_unknown[action_index,i,j]

def objective_function(population_instace):
	lamda = 2

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose, previous_pose

	# loss = npy.zeros(shape=(transition_space,transition_space))
	# alpha = 0.01
	# lamda = 1.
	# w = transition_space/2
	# temp = 0.
	# for m in range(-w,w+1):
	# 	for n in range(-w,w+1):
	# 		# loss[w+m,w+n] = target_belief[previous_pose[0]+m,previous_pose[1]+n] - to_state_belief[previous_pose[0]+m,previous_pose[1]+n]
	# 		loss[w+m,w+n] = target_belief[previous_pose[0]+m,previous_pose[1]+n] - to_state_belief[current_pose[0]+m,current_pose[1]+n]
	# 		temp = trans_mat_unknown[action_index,w+m,w+n] + alpha * loss[w+m,w+n]
	# 		# trans_mat_unknown[action_index,w+m,w+n] += alpha * loss[w+m,w+n]
	# 		if (temp<=1)and(temp>=0):
	# 			trans_mat_unknown[action_index,w+m,w+n]=temp					
	# trans_mat_unknown[action_index] /= trans_mat_unknown[action_index].sum()

	total_loss = 0.

	for m in range(0,discrete_size):	
		for n in range(0,discrete_size):
			total_loss+= target_belief[m,n]-to_state_belief[m,n]

	total_loss += lamda * trans_mat_unknown[action_index,:,:].sum()
	return total_loss

def eval_function_values():
	# sds
	global function_values, population, rri_children, neighbour_children, lc_children
	for i in range(0,pop_size):
		function_values[i] = objective_function(population[i])
		function_values[i+pop_size] = objective_function(neighbour_children[i])
		function_values[i+pop_size*2] = objective_function(rri_children[i])
		function_values[i+pop_size*3] = objective_function(lc_children[i])
	
	
def update_population():
	# master_population = npy.reshape(population,(pop_size*4,var_size))
	global pop_size,population, master_function_values, rri_children, lc_children, neighbour_children
	master_population = npy.zeros(shape=(pop_size*4,var_size))

	for j in range(0,pop_size):
		# for k in range(0,pop_size):
		master_population[j,:]=population[j,:]
		master_population[pop_size+j,:]=neighbour_children[j,:]
		master_population[pop_size*2+j,:]=rri_children[j,:]
		master_population[pop_size*3+j,:]=lc_children[j,:]

	# max_vals = heapq.nlargest(pop_size, master_function_values) 
	# max_args = heapq.nlargest(pop_size, range(len(function_values)), master_function_values.take)
	max_args = heapq.nsmallest(pop_size, range(len(function_values)), function_values.take)
	# max_args = heapq.nsmallest(pop_size, function_values) #, master_function_values.take)
	print "Functions selected:",max_args

	for i in range(0,pop_size):
		population[i,:]=master_population[max_args[i],:]

npy.set_printoptions(precision=2)

def stochastic_search():
	max_iter = 100
	initialize_pop()
	print "Initial population:",population
	eval_function_values()
	
	# calc_mod_values()

	for i in range(0,max_iter):
		print "ITERATION: ",i
		calc_mod_values()
		calc_rel_weights()
		calc_neighbour_children()
		calc_lc_children()
		calc_rri_children()
		eval_function_values()
		print "Function values:", function_values #function_values[0:pop_size]	
		update_population()

# stochastic_search()
# print population

def initialize_state():
	global current_pose, from_state_belief, previous_pose
	from_state_belief[:,:]=0.
	from_state_belief[24,24]=1.
	# from_state_belief[25,24]=0.8

	current_pose=[24,24]
	previous_pose = [24,24]

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
	
	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

	print "Transition Matrices:\n",trans_mat

	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])

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
	from_state_belief[:,:] = dummy[:,:]/dummy.sum()

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
	# for i in range(current_pose[0]-5,current_pose[0]+5):
	# 	print from_state_belief[i,current_pose[1]-5:current_pose[1]+5]
	for i in range(previous_pose[0]-5,previous_pose[0]+5):
		print from_state_belief[i,previous_pose[1]-5:previous_pose[1]+5]
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
	
	to_state_belief[:,:] = dummy[:,:]/dummy.sum()

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
	global trans_mat, from_state_belief, current_pose

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
		# print "No action."		
	# else:
	if (bucket_index!=((transition_space**2)/2)):
		previous_pose = current_pose
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

def back_prop(action_index):
	# global trans_mat_unknown
	# global to_state_belief
	# global from_state_belief
	# global target_belief

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief	

	loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.1

	lamda = 1.

	w = transition_space/2

	delta = 0.
	for ai in range(-w,w+1):
		for aj in range(-w,w+1):
			
			loss[w+ai,w+aj] += lamda * (trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
			
			for i in range(0,discrete_size-2):
				for j in range(0,discrete_size-2):

					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj])
					# delta = (trans_mat_unknown[action_index,:,:].sum()-1.) * trans_mat_unknown[action_index,w+ai,w+aj]
					loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j]) 

					#*(from_state_belief[w+i-ai,w+j-aj]) #+ delta
					# loss[w+ai,w+aj] -= 2*(target_belief[i,j]-to_state_belief[i,j])*(from_state_belief[w+i-ai,w+j-aj]) #+ delta						
					
			# trans_mat_unknown[action_index,w+ai,w+aj] += alpha * loss[w+ai,w+aj]
			trans_mat_unknown[action_index,w+ai,w+aj] -= alpha * loss[w+ai,w+aj]
			# if (trans_mat_unknown[action_index,w+ai,w+aj]<0):
			# 	trans_mat_unknown[action_index,w+ai,w+aj]=0
			# trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()
	trans_mat_unknown[action_index] /=trans_mat_unknown[action_index].sum()

def difference_grad(action_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose, previous_pose

	loss = npy.zeros(shape=(transition_space,transition_space))
	alpha = 0.01
	lamda = 1.
	w = transition_space/2
	temp = 0.
	for m in range(-w,w+1):
		for n in range(-w,w+1):
			# loss[w+m,w+n] = target_belief[previous_pose[0]+m,previous_pose[1]+n] - to_state_belief[previous_pose[0]+m,previous_pose[1]+n]
			loss[w+m,w+n] = target_belief[previous_pose[0]+m,previous_pose[1]+n] - to_state_belief[current_pose[0]+m,current_pose[1]+n]
			temp = trans_mat_unknown[action_index,w+m,w+n] + alpha * loss[w+m,w+n]
			# trans_mat_unknown[action_index,w+m,w+n] += alpha * loss[w+m,w+n]
			if (temp<=1)and(temp>=0):
				trans_mat_unknown[action_index,w+m,w+n]=temp				

	trans_mat_unknown[action_index] /= trans_mat_unknown[action_index].sum()

def recurrence():
	global from_state_belief,target_belief
	from_state_belief = target_belief

def copy_back(action_index):
	global trans_mat_unknown, population
	trans_mat_unknown[action_index,0,:] = population[0:transition_space]
	trans_mat_unknown[action_index,1,:] = population[transition_space,2*transition_space]
	trans_mat_unknown[action_index,2,:] = population[2*transition_space,3*transition_space]

def master(action_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	# belief_prop(action_index)
	# # bayes_obs_fusion()
	# simulated_model(action_index)
	# back_prop(action_index)
	# recurrence()	

	###Fiddling with the order: 
	
	# bayes_obs_fusion()
	# display_beliefs()
	
	belief_prop(action_index)	
	# back_prop(action_index)	
	simulated_model(action_index)	
	# difference_grad(action_index)
	# stochastic_search(action_index)
	stochastic_search()
	copy_back(action_index)
	
	recurrence()	
	
	# print "Transition Matrix: ",action_index,"\n"
	# print trans_mat_unknown[action_index,:,:]
	# print npy.flipud(npy.fliplr(trans_mat_unknown[action_index,:,:]))

initialize_all()

def input_actions():
	global action, state_counter, action_index, current_pose, previous_pose

	# while (action!='q'):		
	iterate=0

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

		if ((current_pose[0]>=48)or(current_pose[0]<=1)or(current_pose[1]>=48)or(current_pose[1]<=1)):
			initialize_state()

		print "Iteration:",iterate," Current pose:",current_pose," Action:",action_index

		master(action_index)

input_actions()

print trans_mat_unknown


