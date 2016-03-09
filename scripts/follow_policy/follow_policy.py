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
# ax.plot_surface(X,Y,path_plot,cmap=plt.cm.jet,cstride=1,rstride=1)
max_number_demos = 50

trajectory_lengths = npy.zeros(max_number_demos)

state_counter = 0
number_demos = 0

# trajectories = [[[0,0],[1,2],[3,4]]]

basis_size=3
# basis_functions = npy.loadtxt(str(sys.argv[1]))
# basis_functions = basis_functions.reshape((basis_size,discrete_size,discrete_size))
basis_functions = npy.zeros(shape=(basis_size,discrete_size,discrete_size))

# reward_weights = npy.loadtxt(str(sys.argv[2]))
# reward_weights = npy.zeros(basis_size)

optimal_policy = npy.loadtxt(str(sys.argv[1]))
optimal_policy = optimal_policy.astype(int)
# reward_function = reward_weights[0]*basis_functions[0]+reward_weights[1]*basis_functions[1]+reward_weights[2]*basis_functions[2]
reward_function = npy.loadtxt(str(sys.argv[2]))
value_function = npy.loadtxt(str(sys.argv[3]))/1000	
# reward_function=npy.zeros(shape=(discrete_size,discrete_size))
path_plot = copy.deepcopy(reward_function)
max_val = npy.amax(path_plot)

max_val_location = npy.unravel_index(npy.argmax(reward_function),reward_function.shape)


# action_space = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........

def follow_policy():

	counter=0	
	
	ax = random.randrange(0,discrete_space_x)
	ay = random.randrange(0,discrete_space_y)

	current_pose[0] = ax
	current_pose[1] = ay
	next_pose=copy.deepcopy(current_pose)
	dummy='y'
	# while (counter<max_path_length)and(dummy=='y'):
	while (counter<max_path_length)and(current_pose!=max_val_location):

		path_plot[current_pose[0]][current_pose[1]]=-max_val/3

		next_pose[0] = current_pose[0] + action_space[optimal_policy[current_pose[0],current_pose[1]]][0]
		next_pose[1] = current_pose[1] + action_space[optimal_policy[current_pose[0],current_pose[1]]][1]

		imshow(path_plot, interpolation='nearest', origin='lower', extent=[0,10,0,10], aspect='auto')
		plt.show(block=False)
		colorbar()
		draw()
		show() 
		current_pose[0] = next_pose[0]		
		current_pose[1] = next_pose[1]		
		counter+=1

		# dummy = raw_input("Continue? ")
follow_policy()

# dummy = npy.amax(reward_function)
# reward_function[3:6,6] = -dummy/4
# reward_function[10,6] = -dummy/4

imshow(optimal_policy, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

imshow(reward_function, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
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

# import numpy as np

n = 50
X, Y = np.mgrid[0:n, 0:n]
# T = np.arctan2(Y - n / 2., X - n/2.)
# R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
# U, V = R * np.cos(T), R * np.sin(T)
# U = action_space[optimal_policy[:,:]][0]
# V = action_space[optimal_policy[:,:]][1]


U = npy.zeros(shape=(discrete_size,discrete_size))
V = npy.zeros(shape=(discrete_size,discrete_size))
x = npy.zeros(shape=(discrete_size,discrete_size))
y = npy.zeros(shape=(discrete_size,discrete_size))

for i in range(0,discrete_size):
	for j in range(0,discrete_size):
		# x[i,j] = action_space[optimal_policy[i,j]][0]
		# y[i,j] = action_space[optimal_policy[i,j]][1]

		U[i,j] = action_space[optimal_policy[i,j]][0]
		V[i,j] = action_space[optimal_policy[i,j]][1]
		# U[i,j]=x[i,j]/(abs(x[i,j])+abs(y[i,j]))
		# V[i,j]=y[i,j]/(abs(x[i,j])+abs(y[i,j]))

# pl.axes([0.025, 0.025, 0.95, 0.95])
print U,V

# # pl.quiver(X, Y, U, V, 10, alpha=.5)
# pl.quiver(X,Y,U,V,reward_function,alpha=1)
# pl.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.1)

# pl.xlim(0, n)
# pl.xticks(())
# pl.ylim(0, n)
# pl.yticks(())

# pl.show() 

fig, ax = plt.subplots()
im = ax.imshow(reward_function, origin='lower',extent=[0,50,0,50])
# im = ax.imshow(value_function, extent=[0,50,0,50])
ax.quiver(V,U)

fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()