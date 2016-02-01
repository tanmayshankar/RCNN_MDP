#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import rospy
# from std_msgs.msg import String
# import roslib
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import random
from scipy.stats import rankdata
from matplotlib.pyplot import *
from scipy import signal
import copy


trans = npy.ones(shape=(3,3))/9
target = npy.array([[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]])
new_weights = npy.ones(shape=(3,3))/9

alpha = 0.1
action = 'y'

while (action!='q'):
	gradient = target - trans
	new_weights = trans + alpha*gradient

	for i in range(0,3):
		for j in range(0,3):
			if (new_weights[i,j]<0.):
				new_weights[i,j]=0.
	if (new_weights.sum()<1.):
		new_weights[:,:]=new_weights[:,:]/new_weights.sum()

	print "Trans:\n",trans
	print "Gradient:\n",gradient
	print "Updated:\n",new_weights

	trans = new_weights
	action = raw_input("Hit a key now: ")