# RCNN_MDP

This repository is for exploring the connection between Markov Decision Processes and Recurrent Convolutional Neural Networks. 

This code base targets 3 problems: 

1. Solving Value / Policy Iteration in a standard MDP using Feedforward passes of an RCNN. 
- Representing the Bayes Filter state belief update as feedforward passes of an RCNN. 
- Learning the State Transition models and Observation model in a POMDP/PODRL setting from simulations.
- Learning Reward Functions in an Inverse Reinforcement Learning framework from demonstrations.  

Problem 1 may be addressed by running the appropriate script, with a stationary reward function as argument. Here's an example: 

./scripts/feedforward_rcnn/rcnn_mdp_value_iteration.py data/trial_3/reward_function.txt

If you'd like to run it with a different transition function of your choice: 
./scripts/feedforward_rcnn/variable_transition_size.py data/trial_3/reward_function.txt

Once either of the feedforward passes are run, you may display the policy, reward and value functions by running the following:

./scripts/display/display_policy.py output_policy.txt reward_function.txt value_function.txt

Currently, you may observe the outputs of learning the transition probabilities by running any of the following codes:

./scripts/belief_prop_rcnn/calc_trans.py

To replan (execute value iteration) with the learnt transition model, run: 

./scripts/feedforward_rcnn/learnt_trans_feedforward.py reward_function.txt estimated_transition.txt
