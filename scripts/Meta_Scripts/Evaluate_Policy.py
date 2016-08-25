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

for i in range(1,26):
	orig_policy = npy.loadtxt("reward_{0}/NEW/output_policy_beta.txt".format(i))

	learnt_policy_po = npy.loadtxt("reward_{0}/NEW/output_policy_linear.txt".format(i))
	replan_error[0,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_po))/ 25

	learnt_policy_fo = npy.loadtxt("reward_{0}/NEW/output_policy_FO_beta.txt".format(i))
	replan_error[1,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_fo))/ 25
