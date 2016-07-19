#!/usr/bin/env python

from variables import *

trans_grad_accum = npy.zeros((action_size,transition_space,transition_space))
trans_filter_grad_accum = npy.zeros(action_size)
epsilon = 0.00001

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

def initialize_observation():
	global observation_model
	observation_model = npy.array([[0.,0.05,0.],[0.05,0.8,0.05],[0.,0.05,0.]])
	# observation_model = npy.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
	
	epsilon=0.0001
	observation_model += epsilon
	observation_model /= observation_model.sum()

def bayes_obs_fusion():
	global to_state_belief, current_pose, observation_model, obs_space, observed_state, corr_to_state_belief, norm_sum_bel
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))
	h = obs_space/2
	for i in range(-h,h+1):
		for j in range(-h,h+1):
			dummy[observed_state[0]+i,observed_state[1]+j] = to_state_belief[observed_state[0]+i,observed_state[1]+j] * observation_model[h+i,h+j]
	corr_to_state_belief[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())
	norm_sum_bel = dummy.sum()

def remap_indices(dummy_index):

	#####action_space = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
	#####    UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT..

	if (dummy_index==0):
		return 4
	if (dummy_index==1):
		return 0
	if (dummy_index==2):
		return 5
	if (dummy_index==3):
		return 2	
	if (dummy_index==5):
		return 3
	if (dummy_index==6):
		return 6
	if (dummy_index==7):
		return 1
	if (dummy_index==8):
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

def initialize_obs_model_bucket():
	global obs_bucket_space, observation_model, obs_space, obs_cummulative
	for i in range(0,obs_space):
		for j in range(0,obs_space):
			obs_cummulative += observation_model[i,j]
			obs_bucket_space[obs_space*i+j] = obs_cummulative

	print obs_bucket_space

def initialize_all():
	initialize_state()
	initialize_observation()
	initialize_transitions()
	initialize_unknown_transitions()
	initialize_model_bucket()
	initialize_obs_model_bucket()

def construct_from_ext_state():
	global from_state_ext, from_state_belief,discrete_size
	d=discrete_size
	from_state_ext[w:d+w,w:d+w] = copy.deepcopy(from_state_belief[:,:])

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

	to_state_belief[:,:] = copy.deepcopy(to_state_ext[w:d+w,w:d+w])

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

	remap_index = remap_indices(bucket_index)
	
	if (bucket_index!=((transition_space**2)/2)):
		current_pose[0] += action_space[remap_index][0]
		current_pose[1] += action_space[remap_index][1]
				
	target_belief[:,:] = 0. 
	target_belief[current_pose[0],current_pose[1]]=1.
	 
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

def belief_prop(action_index):
	global trans_mat_unknown, to_state_belief, from_state_belief	

	to_state_belief = signal.convolve2d(from_state_belief,trans_mat_unknown[action_index],'same','fill',0)
	if (to_state_belief.sum()<1.):
		to_state_belief /= to_state_belief.sum()

def calc_intermed_bel():
	# global to_state_belief, current_pose, observation_model, obs_space, observed_state, corr_to_state_belief, norm_sum_bel
	global to_state_belief, target_belief, observation_model, obs_space, observed_state, intermed_bel
	
	dummy = npy.zeros(shape=(discrete_size,discrete_size))
	h = obs_space/2

	for i in range(-h,h+1):
		for j in range(-h,h+1):
			dummy[observed_state[0]+i,observed_state[1]+j] = (target_belief[observed_state[0]+i,observed_state[1]+j] - to_state_belief[observed_state[0]+i,observed_state[1]+j]) * observation_model[h+i,h+j]
	intermed_bel[:,:] = copy.deepcopy(dummy[:,:]/dummy.sum())

def calc_sensitivity():
	global from_state_ext, sens_belief

	sens_belief = copy.deepcopy(from_state_ext)
	sens_belief = npy.fliplr(sens_belief)
	sens_belief = npy.flipud(sens_belief)

def back_prop_conv_RMS(action_index, time_index):
	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, lamda_vector, sens_belief

	calc_intermed_bel()
	calc_sensitivity()

	w = transition_space/2
	time_count[action_index] +=1
	# alpha = learning_rate - annealing_rate*time_count[action_index]

	grad_update = -signal.convolve2d(sens_belief, intermed_bel, 'valid')
	grad_update[:,:] += lamda*(trans_mat_unknown[action_index,:,:].sum() - 1.)

	penalty_sub = 2
	penalty_super = 2

	# Linear penalties
	for m in range(-w,w+1): 
		for n in range(-w,w+1): 
			if (trans_mat_unknown[action_index,w+m,w+n]<0): 
				grad_update[w+m,w+n] -= penalty_sub
			if (trans_mat_unknown[action_index,w+m,w+n]>1): 
				grad_update[w+m,w+n] += penalty_super 

	# # Proportional penalties.
	# for m in range(-w,w+1):
	# 	for n in range(-w,w+1):
	# 		if (trans_mat_unknown[action_index,w+m,w+n]<0): 
	# 			grad_update[w+m,w+n] -= penalty_sub * trans_mat_unknown[action_index,w+m,w+n]	
	# 		if (trans_mat_unknown[action_index,w+m,w+n]>1): 
	# 			grad_update[w+m,w+n] += penalty_super * (trans_mat_unknown[action_index,w+m,w+n]-1)

	# # Simultaneous proportional penalties. 
	# for m in range(-w,w+1): 
	# 	for n in range(-w,w+1): 
	# 		grad_update[w+m,w+n]+= penalty_super*(max(trans_mat_unknown[action_index,w+m,w+n],1)-1)-penalty_sub*min(trans_mat_unknown[action_index,w+m,w+n],0)

	trans_filter_grad_accum[action_index] = rms_decay * trans_filter_grad_accum[action_index] + (1-rms_decay) * ((grad_update[:,:]**2)).sum()
	trans_mat_unknown[action_index,:,:] -= learning_rate * grad_update[:,:] / npy.sqrt(trans_filter_grad_accum[action_index]+epsilon)

	# trans_mat_unknown[action_index] -= alpha*grad_update

def recurrence():
	global from_state_belief,target_belief
	from_state_belief = copy.deepcopy(target_belief)

def master(action_index, time_index):

	global trans_mat_unknown, to_state_belief, from_state_belief, target_belief, current_pose

	construct_from_ext_state()
	belief_prop_extended(action_index)
	bayes_obs_fusion()

	simulated_model(action_index)
	simulated_observation_model()

	back_prop_conv_RMS(action_index, time_index)
	recurrence()	

initialize_all()

def input_actions():
	global action, state_counter, action_index, current_pose

	iterate=0

	while (iterate<=time_limit):		
		iterate+=1
		action_index=iterate%8
		print "Iteration:",iterate," Current pose:",current_pose,"Observed State:",observed_state," Action:",action_index
		master(action_index, iterate)

input_actions()

def flip_trans_again():
	for i in range(0,action_size):
		trans_mat_unknown[i] = npy.fliplr(trans_mat_unknown[i])
		trans_mat_unknown[i] = npy.flipud(trans_mat_unknown[i])

flip_trans_again()

print "Learnt Transition Model:\n", trans_mat_unknown

for i in range(0,8):
	trans_mat_unknown[i,:,:] /= trans_mat_unknown[i,:,:].sum()
print "Normalized Transition Model:\n",trans_mat_unknown	

print "Actual Transition Model:\n" , trans_mat
	
with file('actual_transition.txt','w') as outfile: 
	for data_slice in trans_mat:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')

with file('estimated_transition.txt','w') as outfile: 
	for data_slice in trans_mat_unknown:
		outfile.write('#Transition Function.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.2f')

