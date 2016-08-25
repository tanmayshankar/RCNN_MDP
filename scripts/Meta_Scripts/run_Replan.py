def movethings(i):
	shutil.move("output_policy.txt","data/BP_Experiments/Replan/reward_{0}/Weighted/output_policy_linear.txt".format(i))
	shutil.move("value_function.txt","data/BP_Experiments/Replan/reward_{0}/Weighted/value_function_linear.txt".format(i))

for i in range(1,26):
	command = "scripts/VI_RCNN/VI_beta_feedforward.py data/BP_Experiments/Replan/reward_{0}/ORIG/reward_{0}.txt data/learnt_models/estimated_trans_weighted.txt".format(i)
	subprocess.call(command.split(),shell=False)	
	movethings(i)	

for i in range(1,26):
    shutil.move("reward_{0}/value_function_po.txt".format(i),"reward_{0}/NEW/value_function_po.txt".format(i))
    shutil.move("reward_{0}/output_policy_po.txt".format(i),"reward_{0}/NEW/output_policy_po.txt".format(i))

    shutil.move("reward_{0}/value_function_fo.txt".format(i),"reward_{0}/NEW/value_function_fo.txt".format(i))
    shutil.move("reward_{0}/output_policy_fo.txt".format(i),"reward_{0}/NEW/output_policy_fo.txt".format(i))

    shutil.move("reward_{0}/value_function_linear.txt".format(i),"reward_{0}/NEW/value_function_linear.txt".format(i))
    shutil.move("reward_{0}/output_policy_linear.txt".format(i),"reward_{0}/NEW/output_policy_linear.txt".format(i))

    shutil.move("reward_{0}/value_function_out_recur.txt".format(i),"reward_{0}/NEW/value_function_out_recur.txt".format(i))
    shutil.move("reward_{0}/output_policy_out_recur.txt".format(i),"reward_{0}/NEW/output_policy_out_recur.txt".format(i))

    shutil.move("reward_{0}/value_function_rms.txt".format(i),"reward_{0}/NEW/value_function_rms.txt".format(i))
    shutil.move("reward_{0}/output_policy_rms.txt".format(i),"reward_{0}/NEW/output_policy_rms.txt".format(i))


# discrete_size = 25

for i in range(1,26):
	orig_policy = npy.loadtxt("reward_{0}/ORIG/output_policy.txt".format(i))
	learnt_policy_po = npy.loadtxt("reward_{0}/Weighted/output_policy_linear.txt".format(i))
	replan_error[i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_po))/ 25
	

	learnt_policy_fo = npy.loadtxt("reward_{0}/NEW/output_policy_FO.txt".format(i))
	replan_error[1,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_fo))/ 25


for i in range(1,26):
	orig_policy = npy.loadtxt("reward_{0}/NEW/output_policy_beta.txt".format(i))
	learnt_policy_po = npy.loadtxt("reward_{0}/NEW/output_policy_PO_beta.txt".format(i))
	replan_error_1[0,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_po))/ 25
	learnt_policy_fo = npy.loadtxt("reward_{0}/NEW/output_policy_FO_beta.txt".format(i))
	replan_error_1[1,i-1] = float(npy.count_nonzero(orig_policy-learnt_policy_fo))/ 25