#!/usr/bin/env python

import shutil

number_rewards = 25

# Command for moving files.
def traj_movethings(ind):
    shutil.move("Trajectories.txt","data/VI_gen_data/reward_{0}/Traj_2/".format(ind))
    shutil.move("Observed_Trajectories.txt","data/VI_gen_data/reward_{0}/Traj_2/".format(ind))
    shutil.move("Actions_Taken.txt","data/VI_gen_data/reward_{0}/Traj_2/".format(ind))

# Command for running Value Iteration.
for i in range(1,number_rewards+1):
	command = "scripts/Follow_Policy/generate_trajectories_extended.py data/VI_gen_data/reward_{0}/gamma_VI/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/learnt_models/actual_transition.txt".format(i)
	subprocess.call(command.split(),shell=False)
	traj_movethings(i)


# Commad for running display. 
for i in range(1,number_rewards+1):                                                                                                                            
	command = "scripts/Display/display_policy.py data/VI_gen_data/reward_{0}/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/VI_gen_data/reward_{0}/value_function.txt".format(i)
	subprocess.call(command.split(),shell=False)