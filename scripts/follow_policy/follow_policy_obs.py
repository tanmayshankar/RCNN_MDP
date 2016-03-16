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

action_size=8
transition_space = 3
# trajectories = [[[0,0],[1,2],[3,4]]]

basis_size=3
# basis_functions = npy.loadtxt(str(sys.argv[1]))
# basis_functions = basis_functions.reshape((basis_size,discrete_size,discrete_size))
basis_functions = npy.zeros(shape=(basis_size,discrete_size,discrete_size))

# reward_weights = npy.loadtxt(str(sys.argv[2]))
# reward_weights = npy.zeros(basis_size)


trans_mat = npy.loadtxt(str(sys.argv[4]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))
# print trans_mat


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
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
############# UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT........


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


# print trans_mat

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
		
		# print "Remap index: ",remap_index, "Action taken: ",action_space[remap_index]		
				# print "Remap index: ",remap_index, "Action taken: ",action_space[remap_index]		

def initialize_observation():
	global observation_model
	observation_model = npy.array([[0.,0.05,0.],[0.05,0.8,0.05],[0.,0.05,0.]])
	# observation_model = npy.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
	# print observation_model

	epsilon=0.0001
	observation_model += epsilon
	observation_model /= observation_model.sum()


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

	# print "Observed State: ", observed_state


def follow_policy():
	global observed_state

	counter=0	
	
	ax = random.randrange(0,discrete_space_x)
	ay = random.randrange(0,discrete_space_y)

	current_pose[0] = ax
	current_pose[1] = ay
	next_pose=copy.deepcopy(current_pose)
	dummy='y'
	act_ind = 0.
	# while (counter<max_path_length)and(dummy=='y'):
	while (counter<max_path_length)and(current_pose!=max_val_location):

		path_plot[current_pose[0]][current_pose[1]]=-max_val/3

		# act_ind = optimal_policy[current_pose[0],current_pose[1]]
		act_ind = optimal_policy[observed_state[0],observed_state[1]]


		# next_pose[0] = current_pose[0] + action_space[optimal_policy[current_pose[0],current_pose[1]]][0]
		# next_pose[1] = current_pose[1] + action_space[optimal_policy[current_pose[0],current_pose[1]]][1]


		simulated_model(act_ind)
		simulated_observation_model()
		next_pose = copy.deepcopy(current_pose)

		imshow(path_plot, interpolation='nearest', origin='lower', extent=[0,10,0,10], aspect='auto')
		plt.show(block=False)
		colorbar()
		draw()
		show() 
		current_pose[0] = next_pose[0]		
		current_pose[1] = next_pose[1]		
		counter+=1

		# dummy = raw_input("Continue? ")
initialize_all()
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