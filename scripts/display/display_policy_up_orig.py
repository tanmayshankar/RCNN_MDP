#!/usr/bin/env python
import numpy as npy
from scipy.stats import truncnorm
import pylab as pl
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import rankdata
from matplotlib.pyplot import *

max_dist = 5
discrete_size = 50

path_plot = npy.zeros((discrete_size,discrete_size))

max_path_length=30
current_pose = [0,0]
max_number_demos = 50

trajectory_lengths = npy.zeros(max_number_demos)

state_counter = 0
number_demos = 0
basis_size=3

basis_functions = npy.zeros(shape=(basis_size,discrete_size,discrete_size))

optimal_policy = npy.loadtxt(str(sys.argv[1]))
optimal_policy = optimal_policy.astype(int)

# dummy = copy.deepcopy(optimal_policy)
# dummy 
dummy = npy.zeros((discrete_size,discrete_size))

reward_function = npy.loadtxt(str(sys.argv[2]))
value_function = npy.loadtxt(str(sys.argv[3]))/1000	

path_plot = copy.deepcopy(reward_function)
max_val = npy.amax(path_plot)

##THE ORIGINAL ACTION SPACE:
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
##THE MODIFIED ACTION SPACE:
# action_space = npy.array([[1,0],[-1,0],[0,-1],[0,1],[1,-1],[1,1],[-1,-1],[-1,1]])
################# UP,  DOWN,  LEFT, RIGHT,UPLEFT,UPRIGHT,DOWNLEFT,DOWNRIGHT ##################

imshow(reward_function, interpolation='nearest', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

imshow(optimal_policy, interpolation='nearest', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

print "Hellu"
imshow(value_function, interpolation='nearest', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

N=50
Y,X = npy.mgrid[0:N,0:N]

U = npy.zeros(shape=(discrete_size,discrete_size))
V = npy.zeros(shape=(discrete_size,discrete_size))

for i in range(0,discrete_size):
	for j in range(0,discrete_size):
		U[i,j] = action_space[optimal_policy[i,j]][0]
		V[i,j] = action_space[optimal_policy[i,j]][1]

		# U[i,j] = action_space[optimal_policy[i,j]][1]
		# V[i,j] = action_space[optimal_policy[i,j]][0]

u = npy.ones((discrete_size,discrete_size))
v = npy.ones((discrete_size,discrete_size))

fig, ax = plt.subplots()
im = ax.imshow(reward_function, extent=[-1,50,-1,50])

ax.quiver(V,U)
# ax.quiver(U,V)
# ax.quiver(X,Y,V,U)
# ax.quiver(X,Y,u,v)

fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()

fig, ax = plt.subplots()
# im = ax.imshow(reward_function, origin='lower',extent=[0,50,0,50])
im = ax.imshow(value_function, extent=[-1,50,-1,50])
# im = ax.imshow(dummy, origin='lower',extent=[-1,50,-1,50])

# ax.quiver(X,Y,V,U)
# ax.quiver(X,Y,u,v)
ax.quiver(V,U)
# ax.quiver(U,V)

fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()