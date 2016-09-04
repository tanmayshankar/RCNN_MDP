#!/usr/bin/env python

import shutil
import os
import numpy as npy 

number_rewards=25
policy_error_po_2 = npy.zeros(25)
# discrete_size = 25

for i in range(1,26):
	orig_policy = npy.loadtxt("reward_{0}/output_policy.txt".format(i))

	learnt_policy_rms = npy.loadtxt("reward_{0}/output_policy_rms.txt".format(i))
	replan_error[0,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_rms))/ 25

	learnt_policy_linear = npy.loadtxt("reward_{0}/output_policy_linear.txt".format(i))
	replan_error[1,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_linear))/ 25

	learnt_policy_po = npy.loadtxt("reward_{0}/output_policy_po.txt".format(i))
	replan_error[2,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_po))/ 25

	learnt_policy_fo = npy.loadtxt("reward_{0}/output_policy_fo.txt".format(i))
	replan_error[3,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_fo))/ 25

	learnt_policy_out = npy.loadtxt("reward_{0}/output_policy_out_recur.txt".format(i))
	replan_error[4,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_out))/ 25

orig_policies = npy.zeros((25,50,50))
learnt_policies = npy.zeros((25,50,50))
replan_error = npy.zeros(25)

for i in range(1,26):
	orig_policies[i-1] = npy.loadtxt("Original/reward_{0}/output_policy.txt".format(i))
	learnt_policies[i-1] = npy.loadtxt("No_Ex_Rep/reward_{0}/Replan/URUT/output_policy.txt".format(i))
	replan_error[i-1] =  float(npy.count_nonzero(orig_policies[i-1]-learnt_policies[i-1]))/ 25



# EVALUATE POLICY BY RUNNING VALUE ITERATION
# DONE FOR UTKR REPLAN PO NEW
# DONE FOR UTUR REPLAN PO NEW
# NOW DOING FOR URKT REPLAN PO NEW

def eval_movethings(i):
	shutil.move("value_function_evaluated.txt","data/QMDP_Experiments/No_Ex_Rep/reward_{0}/Replan/URUT/value_function_evaluated.txt".format(i))

for i in range(1,26):
	command = "scripts/VI_RCNN/VI_evaluate_policy.py data/QMDP_Experiments/RMSProp/reward_{0}/reward_{0}.txt data/learnt_models/actual_transition.txt data/QMDP_Experiments/No_Ex_Rep/reward_{0}/Replan/URUT/output_policy.txt".format(i)
	subprocess.call(command.split(),shell=False)
	eval_movethings(i)
	print "ITERATION:", i

orig_values = npy.zeros((25,50,50))
learnt_values = npy.zeros((25,50,50))
diff_values = npy.zeros((25,50,50))
replan_values = npy.zeros(25)

for i in range(1,26):
	orig_values[i-1] = npy.loadtxt("Original/reward_{0}/orig_policy_value_function_evaluated.txt".format(i))
	learnt_values[i-1] = npy.loadtxt("No_Ex_Rep/reward_{0}/Replan/URKT/value_function_evaluated.txt".format(i))
	diff_values[i-1] = learnt_values[i-1] - orig_values[i-1]
	replan_values[i-1] = float((diff_values[i-1,:,:]/orig_values[i-1,:,:]).sum())/25

for i in range(1,26):
	orig_values[i-1] = npy.loadtxt("Original/reward_{0}/orig_policy_value_function_evaluated.txt".format(i))
	learnt_values[i-1] = npy.loadtxt("No_Ex_Rep/reward_{0}/Replan/URUT/value_function_evaluated.txt".format(i))
	diff_values[i-1] = learnt_values[i-1] - orig_values[i-1]
	replan_values[i-1] = float((diff_values[i-1,:,:]/orig_values[i-1,:,:]).sum())/25

for i in range(1,26):
	orig_policies[i-1] = npy.loadtxt("Original/reward_{0}/output_policy.txt".format(i))
	learnt_policies[i-1] = npy.loadtxt("RMSProp/reward_{0}/Replan/URUT/output_policy.txt".format(i))
	replan_error[i-1] =  float(npy.count_nonzero(orig_policies[i-1]-learnt_policies[i-1]))/ 25