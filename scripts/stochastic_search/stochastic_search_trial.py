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

var_size = 3

decision_variables = npy.zeros(var_size)

min_value = 0.
max_value = +5. 

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
	calc_rri_children()
	population=rri_children

def objective_function(population_instace):
	lamda = 3
	# return (population_instace[0]-0.1)**2+(population_instace[1]-0.1)**2+(population_instace[2]-0.1)**2 ##+ lamda*(population_instace.sum())
	return (population_instace[0]-4)**2+(population_instace[1]-3)**2+(population_instace[2]-2)**2 ##+ lamda*(population_instace.sum())

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

def master():
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

master()

print population








