#!/usr/bin/env python

import shutil

number_rewards = 25

# Command for running Value Iteration.
for i in range(1,number_rewards+1):
	command = "scripts/VI_RCNN/VI_feedforward.py data/VI_gen_data/reward_{0}/max_reward/Max_reward.txt data/learnt_models/actual_transition.txt".format(i)
	subprocess.call(command.split(),shell=False)
	IRL_Replan_movethings(i)

def IRL_Replan_movethings(ind):  
    shutil.move("output_policy.txt","data/VI_gen_data/reward_{0}/ER_no_RMS_Replan/".format(ind))
    shutil.move("value_function.txt","data/VI_gen_data/reward_{0}/ER_no_RMS_Replan/".format(ind))
    shutil.move("Q_Value_Function.txt","data/VI_gen_data/reward_{0}/ER_no_RMS_Replan/".format(ind))

def IRL_Replan_movethings(ind):  
    shutil.move("output_policy.txt","data/VI_gen_data/reward_{0}/ER_po_replan/".format(ind))""
    shutil.move("value_function.txt","data/VI_gen_data/reward_{0}/ER_po_replan/".format(ind))
    shutil.move("Q_Value_Function.txt","data/VI_gen_data/reward_{0}/ER_po_replan/".format(ind))

# Command for running Value Iteration.
for i in range(1,number_rewards+1):
	command = "scripts/VI_RCNN/VI_extended_actions.py data/VI_gen_data/reward_{0}/ER_po_trans/Reward_Function_Estimate.txt data/learnt_models/estimated_trans_po.txt".format(i)
	subprocess.call(command.split(),shell=False)
	IRL_Replan_movethings(i)

# Command for moving files.
def IRL_Replan_movethings(ind):
    shutil.move("output_policy.txt","data/VI_gen_data/reward_{0}/IRL_plan_4/".format(ind))
    shutil.move("value_function.txt","data/VI_gen_data/reward_{0}/IRL_plan_4/".format(ind))
    # shutil.move("Q_Value_Function.txt","data/VI_gen_data/reward_{0}/IRL_plan_4/".format(ind))

# Command for running display. 
for i in range(1,number_rewards+1):                                                                                                                            
	command = "scripts/Display/display_policy.py data/VI_gen_data/reward_{0}/ER_po_replan/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/VI_gen_data/reward_{0}/ER_po_replan/value_function.txt".format(i)
	subprocess.call(command.split(),shell=False)