#!/usr/bin/env python

from variables import *

action_size = 9

q_value_estimate = npy.loadtxt(str(sys.argv[1]))
q_value_estimate = q_value_estimate.reshape((action_size, discrete_size,discrete_size))

def show_image(image_arg):
	plt.imshow(image_arg, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
	plt.show(block=False)
	plt.colorbar()
	plt.show() 

for i in range(0,action_size):
	show_image(q_value_estimate[i])