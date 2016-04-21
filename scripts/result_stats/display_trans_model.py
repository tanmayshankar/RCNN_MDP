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

action_size = 8
transition_space = 3

trans_mat = npy.loadtxt(str(sys.argv[1]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

trans_mat_2 = npy.loadtxt(str(sys.argv[2]))
trans_mat_2 = trans_mat_2.reshape((action_size,transition_space,transition_space))
 
dummy_trans_2 = copy.deepcopy(trans_mat)
dummy_trans_2 -=trans_mat_2

epsilon=0.001
trans_mat+=epsilon
trans_mat_2+=epsilon

trans_mat/=trans_mat.sum()
trans_mat_2/=trans_mat_2.sum()
	
print "Entropy Error:", npy.sum((trans_mat[:,:,:] * npy.log(trans_mat_2[:,:,:])) + ((1-trans_mat[:,:,:])*npy.log(1 - trans_mat_2[:,:,:]) ))
print "Standard Deviation", npy.sqrt(npy.sum(dummy_trans_2[:]**2)/(transition_space**2))	
		
	


