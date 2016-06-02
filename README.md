# RCNN_MDP

This repository is for exploring the connection between Markov Decision Processes, Reinforcement Learning, and Recurrent Convolutional Neural Networks. 

This code base targets 3 problems: 

1. Solving Value / Policy Iteration in a standard MDP using Feedforward passes of a Value Iteration RCNN. 
- Representing the Bayes Filter state belief update as feedforward passes of a Belief Propagation RCNN. 
- Learning the State Transition models in a POMDP setting, using backpropagation on the Belief Propagation RCNN. 
- Learning Reward Functions in an Inverse Reinforcement Learning framework from demonstrations, using a QMDP RCNN. 

**Value Iteration RCNN**

Problem 1 may be addressed by running the appropriate script, with a stationary reward function as argument. Here's an example: 

`./scripts/feedforward_rcnn/rcnn_mdp_value_iteration.py data/trial_3/reward_function.txt`

If you'd like to run it with a different transition function of your choice: 

`./scripts/feedforward_rcnn/variable_transition_size.py data/trial_3/reward_function.txt`

To run Value Iteration with reward as a function of actions as well, run: 

`./scripts/feedforward_rcnn/action_reward.py data/trial_6/reward_function.txt actual_transition.txt`

**Displaying Optimal Policy**

Once either of the feedforward passes are run, you may display the policy, reward and value functions by running the following:

`./scripts/display/display_policy.py output_policy.txt reward_function.txt value_function.txt`

**Learning Transition Model**

To learn the state transition model of the MDP, the following codes are used for various settings. 
For the vanilla version, run: 

`./scripts/belief_prop_rcnn/learn_trans_filter_decay.py`

To run backpropagation as convolution of sensitivies, run: 

`./scripts/conv_back_prop/learn_trans.py`

To learnt the transition model under a partially observable setting, run: 

`./scripts/conv_back_prop/learn_trans_po.py`

To replan (execute value iteration) with the learnt transition model, run: 

`./scripts/feedforward_rcnn/learnt_trans_feedforward.py reward_function.txt estimated_transition.txt`

**Following Optimal Policy**

To watch an agent follow the optimal policy from a random position, with the learnt transition values, run: 

`./scripts/follow_policy/follow_policy_trans.py output_policy.txt reward_function.txt value_function.txt estimated_transition.txt`

**Generating Trajectories using Optimal Policy**

To generate multiple trajectories of an agent following the optimal policy from a random position, run: 

`./scripts/follow_policy/generate_trajectories.py data/trials/VI_trials/trial_8_bounded/output_policy.txt data/trials/VI_trials/trial_8_bounded/output_policy.txt data/learnt_models/actual_transition.txt`

To learn reward functions in an Inverse Reinforcement Learning setting using the QMDP RCNN, the following codes are used. 
For the fully observable setting, run: 

`./scripts/QMDP_IRL/QMDP_selected_traj.py data/learnt_models/actual_transition.txt data/trials/QMDP_trials/trial_15_sync/Trajectories.txt data/trials/QMDP_trials/trial_15_sync/Observed_Trajectories.txt data/trials/QMDP_trials/trial_15_sync/Actions_Taken.txt`

For the partially observable setting, run: 

`./scripts/QMDP_IRL/QMDP_partial_obs.py data/learnt_models/actual_transition.txt data/trials/QMDP_trials/trial_15_sync/Trajectories.txt data/trials/QMDP_trials/trial_15_sync/Observed_Trajectories.txt data/trials/QMDP_trials/trial_15_sync/Actions_Taken.txt`
