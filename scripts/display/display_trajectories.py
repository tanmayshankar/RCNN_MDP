#!/usr/bin/env python

from variables import *

trajectories = npy.loadtxt(str(sys.argv[1]))
trajectories = trajectories.reshape((number_trajectories,trajectory_length,2))

def show_image(image_arg,i):
	plt.imshow(image_arg, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
	plt.show(block=False)
	plt.colorbar()
	plt.title('Trajectory Index: %i' %(i))
	plt.show() 

for i in range(0,number_trajectories):
	path_plot = npy.zeros((discrete_size,discrete_size))
	for j in range(0,trajectory_length):
		path_plot[trajectories[i,j,0],trajectories[i,j,1]] = 100 + j
		show_image(path_plot,i)