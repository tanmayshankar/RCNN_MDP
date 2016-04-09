#!/usr/bin/env python

value_function = npy.zeros((discrete_size,discrete_size))

belief_reward = npy.zeros((action_size,discrete_size,discrete_size))

# for i in range(0,action_size):
# 	belief_reward[i,:,:] = action_reward[i,:,:]*belief[:,:]

