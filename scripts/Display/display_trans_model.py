#!/usr/bin/env python

from variables import *

trans_mat = npy.loadtxt(str(sys.argv[1]))
trans_mat = trans_mat.reshape((action_size,transition_space,transition_space))

def modify_trans_mat():
	global trans_mat
	epsilon = 0.0001
	for i in range(0,action_size):
		trans_mat[i][:][:] += epsilon
		trans_mat[i] /= trans_mat[i].sum()

	for i in range(0,action_size):
		trans_mat[i] = npy.fliplr(trans_mat[i])
		trans_mat[i] = npy.flipud(trans_mat[i])
		
modify_trans_mat()
disp_trans = -npy.ones((transition_space*action_size+action_size-1,transition_space))/2

for i in range(0,action_size):
	for j in range(0,transition_space):
		disp_trans[4*i+j,:] = trans_mat[action_size-i-1,j,:]

imshow(disp_trans, interpolation='nearest', origin='lower', extent=[0,50,0,50], aspect='auto')
plt.show(block=False)
colorbar()
draw()
show() 

