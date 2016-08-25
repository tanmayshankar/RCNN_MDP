#!/usr/bin/env python

import shutil

# Command for running Value Iteration.
for i in range(1,26):
	command = "scripts/VI_RCNN/VI_action_reward.py data/VI_gen_data/reward_{0}/reward_{0}.txt data/learnt_models/actual_transition.txt".format(i)
	subprocess.call(command.split(),shell=False)
	movethings(i)

# Command for moving files.
def movethings(ind):
    shutil.move("action_reward_function.txt","data/VI_gen_data/reward_{0}/".format(ind))
    shutil.move("output_policy.txt","data/VI_gen_data/reward_{0}/".format(ind))
    shutil.move("value_function.txt","data/VI_gen_data/reward_{0}/".format(ind))
    shutil.move("Q_Value_Function.txt","data/VI_gen_data/reward_{0}/".format(ind))

def movethings(ind):
    shutil.move("action_reward_function.txt","data/VI_gen_data/reward_{0}/gamma_VI/".format(ind))
    shutil.move("output_policy.txt","data/VI_gen_data/reward_{0}/gamma_VI/".format(ind))
    shutil.move("value_function.txt","data/VI_gen_data/reward_{0}/gamma_VI/".format(ind))
    shutil.move("Q_Value_Function.txt","data/VI_gen_data/reward_{0}/gamma_VI/".format(ind))


# Commad for running display. 
for i in range(1,26):                                                                                                                            
	command = "scripts/Display/display_policy.py data/VI_gen_data/reward_{0}/gamma_VI/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/VI_gen_data/reward_{0}/gamma_VI/value_function.txt".format(i)
	subprocess.call(command.split(),shell=False)

for i in range(1,26):                                                                                                                            
	command = "scripts/Display/display_policy.py data/VI_gen_data/reward_{0}/IRL_plan_4/output_policy.txt data/VI_gen_data/reward_{0}/reward_{0}.txt data/VI_gen_data/reward_{0}/IRL_plan_4/value_function.txt".format(i)
	subprocess.call(command.split(),shell=False)