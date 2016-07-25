#!/usr/bin/env python

from variables import *

def initialize_state():
	global current_pose, from_state_belief
	from_state_belief[24,24]=1.
	current_pose=[24,24]

def initialize_transitions():
	global trans_mat
	trans_mat_1 = npy.array([[0.,0.7,0.],[0.1,0.1,0.1],[0.,0.,0.]])
	trans_mat_2 = npy.array([[0.7,0.1,0.],[0.1,0.1,0.],[0.,0.,0.]])
	
	#Adding epsilon so that the cummulative distribution has unique values. 
	epsilon=0.001
	trans_mat_1+=epsilon
	trans_mat_2+=epsilon

	trans_mat_1/=trans_mat_1.sum()
 	trans_mat_2/=trans_mat_2.sum()

	trans_mat[0] = trans_mat_1
	trans_mat[1] = npy.rot90(trans_mat_1,2)
	trans_mat[2] = npy.rot90(trans_mat_1,1)
	trans_mat[3] = npy.rot90(trans_mat_1,3)

	trans_mat[4] = trans_mat_2
	trans_mat[5] = npy.rot90(trans_mat_2,3)	
	trans_mat[7] = npy.rot90(trans_mat_2,2)
	trans_mat[6] = npy.rot90(trans_mat_2,1)

def initialize_unknown_transitions():
	global trans_mat_unknown

	for i in range(0,transition_space):
		for j in range(0,transition_space):
			trans_mat_unknown[:,i,j] = random.random()
			# trans_mat_unknown[:,i,j] = 1.
	for i in range(0,action_size):
		trans_mat_unknown[i,:,:] /=trans_mat_unknown[i,:,:].sum()

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

def initialize_model_bucket():
	global cummulative, bucket_index, bucket_space
	orig_mat = copy.deepcopy(trans_mat)
	for k in range(0,action_size):
		orig_mat = npy.flipud(npy.fliplr(trans_mat[k,:,:]))

		for i in range(0,transition_space):
			for j in range(0,transition_space):				
				cummulative[k] += orig_mat[i,j]
				bucket_space[k,transition_space*i+j] = cummulative[k]

def initialize_all():
	initialize_state()	
	initialize_transitions()	
	initialize_unknown_transitions()
	initialize_model_bucket()

def construct_from_ext_state():
	global from_state_ext, from_state_belief,discrete_size
	d=discrete_size
	from_state_ext[w:d+w,w:d+w] = from_state_belief[:,:]	

def belief_prop_extended(action_index):
	global trans_mat_unknown, from_state_ext, to_state_ext, w, discrete_size
	to_state_ext = signal.convolve2d(from_state_ext,trans_mat_unknown[action_index],'same')
	d=discrete_size
	##NOW MUST FOLD THINGS:
	for i in range(0,2*w):
		to_state_ext[i+1,:]+=to_state_ext[i,:]
		to_state_ext[i,:]=0
		to_state_ext[:,i+1]+=to_state_ext[:,i]
		to_state_ext[:,i]=0
		to_state_ext[d+2*w-i-2,:]+= to_state_ext[d+2*w-i-1,:]
		to_state_ext[d+2*w-i-1,:]=0
		to_state_ext[:,d+2*w-i-2]+= to_state_ext[:,d+2*w-i-1]
		to_state_ext[:,d+2*w-i-1]=0

	to_state_belief[:,:] = to_state_ext[w:d+w,w:d+w]

def simulated_model(action_index):
	global trans_mat, from_state_belief, bucket_space, bucket_index, cummulative

	rand_num = random.random()

	if (rand_num<bucket_space[action_index,0]):
		bucket_index=0
	
	for i in range(1,transition_space**2):
		if (bucket_space[action_index,i-1]<=rand_num)and(rand_num<bucket_space[action_index,i]):
			bucket_index=i
	
	remap_index = remap_indices(bucket_index)
	
	if (bucket_index!=((transition_space**2)/2)):
		current_pose[0] += action_space[remap_index][0]
		current_pose[1] += action_space[remap_index][1]
			
	target_belief[:,:] = 0. 
	target_belief[current_pose[0],current_pose[1]]=1.
	
def belief_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief	

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()

def back_prop(action_index,time_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, lamda_vector

	w = transition_space/2

	time_count[action_index] +=1
	alpha = learning_rate - annealing_rate*time_count[action_index]
	
	for m in range(-w,w+1):
		for n in range(-w,w+1):
			loss_1=0.
			for i in range(0,discrete_size):
				for j in range(0,discrete_size):
					if (i-m>=0)and(i-m<discrete_size)and(j-n>=0)and(j-n<discrete_size):
						loss_1 -= 2*(target_belief[i,j]-to_state_belief[i,j])*from_state_belief[i-m,j-n]
			
			loss_1 += lamda_vector[action_index] * (trans_mat_unknown[action_index,:,:].sum() - 1.)
			
			if (trans_mat_unknown[action_index,w+m,w+n] - alpha*loss_1>=0)and(trans_mat_unknown[action_index,w+m,w+n] - alpha*loss_1<1):
				trans_mat_unknown[action_index,w+m,w+n] -= alpha*loss_1

			lamda_vector[action_index] -= alpha * ((trans_mat_unknown[action_index,:,:].sum()-1.)**2)

def calc_sensitivity():
	global from_state_ext, sens_belief

	sens_belief = copy.deepcopy(from_state_ext)

	sens_belief = npy.fliplr(sens_belief)
	sens_belief = npy.flipud(sens_belief)

def back_prop_conv(action_index, time_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, lamda_vector, sens_belief

	calc_sensitivity()

	w = transition_space/2
	time_count[action_index] +=1
	alpha = learning_rate - annealing_rate*time_count[action_index]

	grad_update = signal.convolve2d(sens_belief, target_belief - to_state_belief, 'valid')
	trans_mat_unknown[action_index] += alpha*grad_update

def recurrence():
	global from_state_belief,target_belief
	from_state_belief = copy.deepcopy(target_belief)

def master(action_index, time_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	# belief_prop(action_index)
	construct_from_ext_state()
	belief_prop_extended(action_index)
	simulated_model(action_index)
	back_prop_conv(action_index, time_index)
	recurrence()	

initialize_all()

def input_actions():
	global action, state_counter, action_index, current_pose

	iterate=0

	while (iterate<=time_limit):		
		iterate+=1	
		action_index=iterate%8
		print "Iteration:",iterate," Current pose:",current_pose," Action:",action_index
		master(action_index, iterate)

input_actions()

def flip_trans_again():
	for i in range(0,action_size):
		trans_mat_unknown[i] = npy.fliplr(trans_mat_unknown[i])
		trans_mat_unknown[i] = npy.flipud(trans_mat_unknown[i])

flip_trans_again()

print "Transition Matrix: "
print trans_mat_unknown

for i in range(0,action_size):
	print "Trans Mat Sum:", trans_mat_unknown[i].sum()
	trans_mat_unknown[i,:,:] /=trans_mat_unknown[i,:,:].sum()
print "Normalized:\n",trans_mat_unknown	

print "Actual transition matrix:" , trans_mat

print "Lamda Vector:", lamda_vector

with file('actual_transition.txt','w') as outfile: 
	for data_slice in trans_mat:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')

with file('estimated_transition.txt','w') as outfile: 
	for data_slice in trans_mat_unknown:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')