#!/usr/bin/env python

import shutil

number_rewards = 25

# Command for moving files.
def IRL_movethings(ind):
    shutil.move("Reward_Function_Estimate.txt","data/VI_gen_data/reward_{0}/ER_po_trans/".format(ind))
    shutil.move("Q_Value_Estimate.txt","data/VI_gen_data/reward_{0}/ER_po_trans/".format(ind))
    shutil.move("Value_Function_Estimate.txt","data/VI_gen_data/reward_{0}/ER_po_trans/".format(ind))

# Command for running Value Iteration.
for i in range(1,number_rewards+1):
	command = "scripts/QMDP_RCNN/experience_replay_RMSProp.py data/learnt_models/actual_transition.txt data/VI_gen_data/reward_{0}/Traj_2/Trajectories.txt data/VI_gen_data/reward_{0}/Traj_2/Observed_Trajectories.txt data/VI_gen_data/reward_{0}/Traj_2/Actions_Taken.txt".format(i)
	subprocess.call(command.split(),shell=False)
	IRL_movethings(i)

for i in range(1,number_rewards+1):
	command = "scripts/QMDP_RCNN/experience_replay.py data/learnt_models/actual_transition.txt data/VI_gen_data/reward_{0}/Traj_2/Trajectories.txt data/VI_gen_data/reward_{0}/Traj_2/Observed_Trajectories.txt data/VI_gen_data/reward_{0}/Traj_2/Actions_Taken.txt".format(i)
	subprocess.call(command.split(),shell=False)
	IRL_movethings(i)

for i in range(1,number_rewards+1):
	command = "scripts/QMDP_RCNN/experience_replay.py data/learnt_models/estimated_trans_po.txt data/VI_gen_data/reward_{0}/Traj_1/Trajectories.txt data/VI_gen_data/reward_{0}/Traj_1/Observed_Trajectories.txt data/VI_gen_data/reward_{0}/Traj_1/Actions_Taken.txt".format(i)
	subprocess.call(command.split(),shell=False)
	IRL_movethings(i)

# Commad for running display. 
for i in range(1,number_rewards+1):                                                                                                                            
	command = "scripts/Display/display_policy.py data/VI_gen_data/reward_{0}/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/VI_gen_data/reward_{0}/value_function.txt".format(i)
	subprocess.call(command.split(),shell=False)