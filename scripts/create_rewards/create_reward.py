#!/usr/bin/env python
import numpy as npy
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.stats import rankdata
import math
import shutil
import copy

radius_threshold = 10
discrete_size = 100

discrete_space_x = 50
discrete_space_y = 50
max_dist = 5

action = 'e'
new_demo = 'y'

#Dummy set of variables for the random object spatial querying. 
space_dist = npy.linspace(-max_dist,max_dist,discrete_space_x)

# number_objects = 130
number_objects = 41

rad_dist = npy.linspace(0,radius_threshold,discrete_size)
# print rad_dist

#READING THE spatial distribution function from file. 
pairwise_value_function = npy.loadtxt(str(sys.argv[1]))
# Note that this returned a 2D array!
print pairwise_value_function.shape
pairwise_value_function = pairwise_value_function.reshape((number_objects,number_objects,discrete_size))

value_function = npy.zeros(shape=(discrete_space_x,discrete_space_y))
reward_val = npy.zeros(shape=(discrete_space_x,discrete_space_y))
value_functions = npy.zeros(shape=(number_objects,discrete_space_x,discrete_space_y))
# number_objects = 41
object_confidence = npy.zeros(number_objects)
object_poses = npy.zeros(shape=(number_objects,2))

# number_objects = 41
object_location_function = npy.zeros(shape=(discrete_space_x,discrete_space_y))

def random_obj_positions():
	number_scene_objects = 30
	xbucket=0
	ybucket=0
	for i in range(0,number_scene_objects):		
		trial_label = random.randrange(0,number_objects)
		object_confidence[trial_label] = random.random()

	for i in range(0,number_objects):
		object_poses[i][0] = random.randrange(-max_dist,max_dist)
 		object_poses[i][1] = random.randrange(-max_dist,max_dist)		

 		if object_poses[i][0]<space_dist[0]:
			xbucket=0
		elif object_poses[i][0]>space_dist[len(space_dist)-1]:
			xbucket=len(space_dist)-1
		else:
			for j in range(0,len(space_dist)):	
				if object_poses[i][0]>space_dist[j] and object_poses[i][0]<space_dist[j+1]:
					xbucket=j

		if object_poses[i][1]<space_dist[0]:
			ybucket=0
		elif object_poses[i][1]>space_dist[len(space_dist)-1]:
			ybucket=len(space_dist)-1
		else:
			for j in range(0,len(space_dist)):	
				if object_poses[i][1]>space_dist[j] and object_poses[i][0]<space_dist[j+1]:
					ybucket=j

		object_location_function[xbucket][ybucket] = object_confidence[i]

random_obj_positions()

#REMEMBER, this is outside the lookup_value_add function. 
def lookup_value_add(sample_pt, obj_find_index):
	value_lookup=0
	for alt_obj in range(0,number_objects):	
		
		rad_value = (((sample_pt[0]-object_poses[alt_obj][0])**2)+((sample_pt[1]-object_poses[alt_obj][1])**2))**0.5
		if rad_value<rad_dist[0]:
			bucket=0;
		elif rad_value>rad_dist[len(rad_dist)-1]:
			bucket=len(rad_dist)-1
		else:
			for i in range(0,len(rad_dist)):	
				if rad_value>rad_dist[i] and rad_value<rad_dist[i+1]:
					bucket=i
		# print sample_pt, rad_value, bucket

		value_lookup += pairwise_value_function[obj_find_index][alt_obj][bucket] * object_confidence[alt_obj] / number_objects
	return value_lookup

def calculate_value_function(obj_find_index):	
	for i in range(0,discrete_space_x): 		
		for j in range(0,discrete_space_y):						
			sample = space_dist[i],space_dist[j]		
			x = lookup_value_add(sample, obj_find_index)			
			value_functions[obj_find_index][i][j]=x

def launch():
	calculate_value_function(4)
	calculate_value_function(1)
	calculate_value_function(3)

	weights = npy.ones(3)
	weights[0] = 0.7
	weights[1] = 0.5
	weights[2] = 0.3
	weights=weights[:]/weights.sum()

	reward_val = (value_functions[4])*weights[0] + (value_functions[1])*weights[1] + (value_functions[3])*weights[2] #- costmap*weights[3]

	# plt.imshow(reward_val, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
	# plt.colorbar()
	# plt.show()

def create_rewards():
	for i in range(7,11):
		print "Iteration: ",i
		launch()
		with file('reward_function_new.txt','w') as outfile:
			outfile.write('#Reward Function.\n')
			npy.savetxt(outfile,100*reward_val,fmt='%-7.4f')
		shutil.move("reward_function_new.txt","data/Rewards/rewards_3/reward_%d.txt" %i)

create_rewards()
