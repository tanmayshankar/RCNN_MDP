#!/usr/bin/env python

import shutil

def movethings(ind):
    shutil.move("action_reward_function.txt","data/Rewards/reward_{0}".format(ind))
    shutil.move("output_policy.txt","data/Rewards/reward_{0}".format(ind))
    shutil.move("value_function.txt","data/Rewards/reward_{0}".format(ind))
    shutil.move("Q_Value_Function.txt","data/Rewards/reward_{0}".format(ind))

for i in range(1,26):
   %run scripts/VI_RCNN/VI_action_reward.py 'data/Rewards/reward_{0}/reward_{0}.txt'.format(i) data/learnt_models/actual_transition.txt
   print i 
   movethings(i)

