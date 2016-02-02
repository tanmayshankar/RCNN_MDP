#!/usr/bin/env python
import numpy as npy
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import rospy
import pylab as pl
from std_msgs.msg import String
import roslib
from nav_msgs.msg import Odometry
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import rankdata
from matplotlib.pyplot import *

max_dist = 5

discrete_space_x = 50
discrete_space_y = 50
discrete_size = 50

path_plot = npy.zeros(shape=(discrete_space_x,discrete_space_y))

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

reward_function = npy.loadtxt(str(sys.argv[2]))
value_function = npy.loadtxt(str(sys.argv[3]))/1000	

path_plot = copy.deepcopy(reward_function)
max_val = npy.amax(path_plot)

# action_space = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........


imshow(reward_function, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

imshow(optimal_policy, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

print "Hellu"
imshow(value_function, interpolation='nearest', origin='lower', extent=[0,10,0,10], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

n = 50
X, Y = np.mgrid[0:n, 0:n]

U = npy.zeros(shape=(discrete_size,discrete_size))
V = npy.zeros(shape=(discrete_size,discrete_size))
x = npy.zeros(shape=(discrete_size,discrete_size))
y = npy.zeros(shape=(discrete_size,discrete_size))

for i in range(0,discrete_size):
	for j in range(0,discrete_size):
		
		U[i,j] = action_space[optimal_policy[i,j]][0]
		V[i,j] = action_space[optimal_policy[i,j]][1]		

# print U,V

fig, ax = plt.subplots()
im = ax.imshow(reward_function, origin='lower',extent=[0,50,0,50])
# im = ax.imshow(value_function, origin='lower',extent=[0,50,0,50])
ax.quiver(V,U)
fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()

fig, ax = plt.subplots()
# im = ax.imshow(reward_function, origin='lower',extent=[0,50,0,50])
im = ax.imshow(value_function, origin='lower',extent=[0,50,0,50])
ax.quiver(V,U)
fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()