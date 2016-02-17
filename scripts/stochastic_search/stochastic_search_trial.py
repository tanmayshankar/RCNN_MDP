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
max_value = 1. 

pop_size = 20
population = npy.zeros(shape=(pop_size,var_size))

# function_values = npy.zeros(shape=(4,pop_size))
master_function_values = npy.zeros(4*pop_size)
mod_func_values = npy.zeros(pop_size)
rel_weights = npy.zeros(pop_size)

sigma = 0.1

# def calculate_objective_function(pop_index):
def calc_mod_values():
	global mod_func_values, function_values
	# mod_func_values = function_values[0,:] - npy.amin(function_values[0,:])
	mod_func_values = function_values[0:pop_size] - npy.amin(function_values[0:pop_size])

def calc_rel_weights():
	global rel_weights,mod_func_values
	# rel_weights = mod_func_values / mod_func_values.sum()
	rel_weights[:] = mod_func_values[:] / mod_func_values.sum()

neighbour_children = npy.zeros(shape=(pop_size,var_size))
num_pop_children = npy.zeros(pop_size)
num_pop_children = num_pop_children.astype(int)

def calc_neighbour_children():
	global neighbour_children, population
	
	# for i in range(0,pop_size):
	# 	rand_num = random.random()
	# 	pop_children[i,:] = population[i,:] + sigma*rand_num

	# for i in range(0,pop_size):
	# 	for j in range(0,var_size):
	# 		rand_num = random.random()
	# 		pop_children[i,j] = population[i,j] + rand_num*sigma

	num_pop_children[:] = rel_weights[:] * pop_size
	# num_pop_children /= num_pop_children.sum()
	num_pop_children = num_pop_children.astype(int)
	child_counter = 0.

	for i in range(0,pop_size):
		for j in range(0,num_pop_children[i]+1):
			for k in range(0,var_size):
				rand_num = random.random()
				neighbour_children[child_counter,k] = rand_num * sigma + population[i,k]
			child_counter+=1

	# for i in range(0,pop_size):
	# 	for j in range(0,num_pop_children[i]+1):
	# 		rand_num = random.random()
	# 		# for k in range(0,var_size):	
	# 			# neighbour_children[child_counter,k] = rand_num * sigma + population[i,k]
	# 		neighbour_children[child_counter,:] = rand_num*sigma + population[i,:]
	# 		child_counter+=1

lc_children = npy.zeros(shape=(pop_size,var_size))

def calc_lc_children():
	global population, lc_children, pop_size

	for i in range(0,pop_size):
		rand_wt = random.random()
		rand_ind_1 = random.randrange(0,pop_size)
		rand_ind_2 = random.randrange(0,pop_size)
		lc_children[i,:] = rand_wt*population[rand_ind_1,:] + (1-rand_wt)*population[rand_ind_2,:]

rri_children = npy.zeros(shape=(pop_size,var_size))

def calc_rri_children():
	global population,rri_children, pop_size

	for i in range(0,pop_size):
		for j in range(0,var_size):
			rand_num=random.random()
			rri_children[i,j]= min_value + rand_num * (max_value-min_value)

def initialize_pop():
	global population,rri_children,pop_size
	calc_rri_children()
	population=rri_children

def eval_function_values(pop_index):
	# sds
	global function_values, population, rri_children, neighbour_children, lc_children

max_population = npy.zeros(shape=(pop_size,var_size))

def update_population():
	# master_population = npy.reshape(population,(pop_size*4,var_size))
	master_population = npy.zeros(shape=(pop_size*4,var_size))

	for i in range(0,4):
		for j in range(0,pop_size):
			# for k in range(0,pop_size):
			master_population[j,:]=population[j,:]
			master_population[pop_size+j,:]=neighbour_children[j,:]
			master_population[pop_size*2+j,:]=rri_children[j,:]
			master_population[pop_size*3+j,:]=lc_children[j,:]

	# max_vals = heapq.nlargest(pop_size, master_function_values) 
	max_args = heapq.nlargest(pop_size, range(len(master_function_values)), master_function_values.take)
	for i in range(0,pop_size):
		population[i,:]=master_population[max_args[i],:]









