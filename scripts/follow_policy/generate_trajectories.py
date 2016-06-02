#!/usr/bin/env python
import numpy as npy
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pylab as pl
# from std_msgs.msg import String
import sys
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import rankdata
from matplotlib.pyplot import *

discrete_size = 50
max_path_length=30
current_pose = [0,0]

max_number_demos = 50
trajectory_lengths = npy.zeros(max_number_demos)

state_counter = 0
number_demos = 0

action_size=8
transition_space = 3
trajectories = [[[0,0],[1,2],[3,4]]]
observed_trajectories = [[[0,0],[1,2],[3,4]]]
actions_taken = [[0,0]]

trans_mat = npy.loadtxt(str(sys.argv[3]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

optimal_policy = npy.loadtxt(str(sys.argv[1]))
optimal_policy = optimal_policy.astype(int)

reward_function = npy.loadtxt(str(sys.argv[2]))
max_val = npy.amax(reward_function)
max_val_location = npy.unravel_index(npy.argmax(reward_function),reward_function.shape)

##THE ORIGINAL ACTION SPACE:
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
##THE MODIFIED ACTION SPACE:
# action_space = npy.array([[1,0],[-1,0],[0,-1],[0,1],[1,-1],[1,1],[-1,-1],[-1,1]])
################# UP,  DOWN,  LEFT, RIGHT,UPLEFT,UPRIGHT,DOWNLEFT,DOWNRIGHT ##################

bucket_space = npy.zeros((action_size,transition_space**2))
cummulative = npy.zeros(action_size)
bucket_index = 0

obs_space=3
observation_model = npy.zeros(shape=(obs_space,obs_space))
obs_model_unknown = npy.ones(shape=(obs_space,obs_space))
observed_state = npy.zeros(2)

obs_bucket_space = npy.zeros(obs_space**2)
obs_bucket_index =0 
obs_cummulative = 0

def modify_trans_mat():
	global trans_mat
	epsilon = 0.0001
	for i in range(0,action_size):
		trans_mat[i][:][:] += epsilon
		trans_mat[i] /= trans_mat[i].sum()

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
	global trans_mat, from_state_belief, bucket_space, bucket_index, cummulative

	#### BASED ON THE TRANSITION MODEL CORRESPONDING TO ACTION_INDEX, PROBABILISTICALLY FIND THE NEXT SINGLE STATE.
	#must find the right bucket

	rand_num = random.random()

	if (rand_num<bucket_space[action_index,0]):
		bucket_index=0
	
	for i in range(1,transition_space**2):
		if (bucket_space[action_index,i-1]<=rand_num)and(rand_num<bucket_space[action_index,i]):
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

	if (current_pose[0]>49):
		current_pose[0]=49
	if (current_pose[1]>49):
		current_pose[1]=49
	if (current_pose[0]<0):
		current_pose[0]=0
	if (current_pose[1]<0):
		current_pose[1]=0
		
		# print "Remap index: ",remap_index, "Action taken: ",action_space[remap_index]		
				# print "Remap index: ",remap_index, "Action taken: ",action_space[remap_index]		

def initialize_observation():
	global observation_model
	observation_model = npy.array([[0.,0.05,0.],[0.05,1.6,0.05],[0.,0.05,0.]])
	# observation_model = npy.array([[0.05,0.05,0.05],[0.05,0.6,0.05],[0.05,0.05,0.05]])
	# observation_model = npy.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
	epsilon=0.0001
	observation_model += epsilon
	observation_model /= observation_model.sum()

	print observation_model

def initialize_obs_model_bucket():
	global obs_bucket_space, observation_model, obs_space, obs_cummulative
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			obs_cummulative += observation_model[i,j]
			obs_bucket_space[obs_space*i+j] = obs_cummulative

	print obs_bucket_space

def initialize_all():
	initialize_observation()
	initialize_obs_model_bucket()
	modify_trans_mat()
	initialize_model_bucket()

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

print optimal_policy

def follow_policy():
	global observed_state, current_pose, trajectory_lengths, trajectories
	state_counter=1	
	demo_counter=1
	
 	new_demo='y'
	act_ind = 0.
	 # number_demos = 1
	max_demo = 50

	# while (new_demo!='n'):
	while (demo_counter<max_demo-2):
	
		ax = random.randrange(0,discrete_size)
		ay = random.randrange(0,discrete_size)

		# ax = 0
		# ay = 0		

		current_pose[0] = ax
		current_pose[1] = ay

		simulated_observation_model()
		
		# current_trajectory = [[ax,ay]]
		current_trajectory = [[current_pose[0], current_pose[1]]]
		current_observed_trajectory = [[observed_state[0],observed_state[1]]] 
		act_ind = optimal_policy[observed_state[0],observed_state[1]]
		current_actions_taken = [act_ind]
		state_counter=1
	
		while (state_counter<max_path_length)and(current_pose!=max_val_location):
			
			simulated_model(act_ind)
			simulated_observation_model()
			
			act_ind = optimal_policy[current_pose[0],current_pose[1]]
			# print "The current pose is:",current_pose
			# print "The observed state is:",observed_state
			# print "Action Taken is:",act_ind

			state_counter+=1

			current_trajectory.append([current_pose[0],current_pose[1]])
			current_observed_trajectory.append([observed_state[0],observed_state[1]])
			current_actions_taken.append(act_ind)

		demo_counter+=1
		print demo_counter

		trajectories.append(current_trajectory)
		observed_trajectories.append(current_observed_trajectory)
		actions_taken.append(current_actions_taken)

		trajectory_lengths[demo_counter] = state_counter
		trajectory_lengths=trajectory_lengths.astype(int)

		# new_demo = raw_input("Do you want to start a new demonstration? ")

initialize_all()
follow_policy()

trajectories.remove(trajectories[0])
observed_trajectories.remove(observed_trajectories[0])
actions_taken.remove(actions_taken[0])

print observed_trajectories

print "The trajectories are as follows: ",trajectories

print "The trans mats are as follows:", trans_mat

with file('Trajectories.txt','w') as outfile:
	# for data_slice in pairwise_value_func:
	for data_slice in trajectories:
		outfile.write('# New slice\n')
		npy.savetxt(outfile,data_slice,fmt='%i')
		
with file('Observed_Trajectories.txt','w') as outfile: 
	
	for data_slice in observed_trajectories:
		outfile.write('#Observed Trajectory.\n')
		npy.savetxt(outfile,data_slice,fmt='%i')

with file('Actions_Taken.txt','w') as outfile:
	for data_slice in actions_taken:
		outfile.write('#Actions Taken.\n')
		npy.savetxt(outfile,data_slice,fmt='%i')