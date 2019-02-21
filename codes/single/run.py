#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:46:14 2019

@author: eriti
"""
from RL_brain import QLearningTable
from train_agent import generatePath, stamp, train_agent
from global_var import green_states, current_time
from verify_single import test_agent, test_plot

if __name__ == '__main__':

    # --------------preparation--------------------
    rst_path, sim_path = generatePath(current_time)
    RL = QLearningTable(list(range(len(green_states))))
    # --------------training--------------------
    train_agent(RL, rst_path, sim_path)
    # --------------verifying--------------------
    RL.epsilon = 1
    fixed, rl, actuated = test_agent('fixed', RL), test_agent('rl', RL), test_agent('actuated', RL)
    test_plot(rl, fixed, actuated, sim_path)
    # ----------------------------------
    save_path = '{}/qtable.csv'.format(sim_path)
    RL.save_qtable(save_path)
    RL.plot_cumulative_reward(sim_path)
    RL_params = {'lr':RL.lr, 'gamma':RL.gamma, 'e_max':RL.e_greedy_max, 'e_inc':RL.e_greedy_increment}
    stamp(RL_params, rst_path, sim_path, clean = True) # need some update, lower priority
    # ----------------------------------
    print('\n ALL DONE, check {}'.format(str(current_time)))

