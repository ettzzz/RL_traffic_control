#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:46:14 2019

@author: eriti
"""
from RL_brain import QLearningTable
from train_agent import generate_path, write_log, train_agent
from global_var import green_states, current_time, WORKSPACE
from verify_single import test_agent, test_plot
from flow_input_single import push_agent


def main():
    # --------------preparation--------------------
    rst_path, sim_path = generate_path(current_time)
    RL = QLearningTable(list(range(len(green_states))))
#    feed_path = '{}/results/{}/qtable.csv'.format(WORKSPACE, '02231101')
#    feed_path = '/Users/eriti/Desktop/Masterarbeit/frompps/results/02160331_095exodus/qtable.csv'
#    RL.feed_qtable(feed_path)
#     --------------training--------------------
    train_agent(RL, rst_path, sim_path)
    # --------------verifying--------------------
    RL.epsilon = 1
    fixed, rl, actuated = test_agent('fixed', RL), test_agent('rl', RL), test_agent('actuated', RL)
    test_plot(rl, fixed, actuated, sim_path)
    flow_scenarios = ['-50%', '-25%',  '0%', '+25%', '+50%']
    push_agent(flow_scenarios, sim_path, RL)
    # ------------------------------------
    RL.save_qtable('{}/qtable.csv'.format(sim_path))
    RL.plot_cumulative_reward(sim_path) # potential bugs
    RL_params = {'lr':RL.lr, 'gamma':RL.gamma, 'e_max':RL.e_greedy_max, 'e_inc':RL.e_greedy_increment}
    write_log(RL_params, rst_path, sim_path, clean = True) # need some update, lower priority
    # ----------------------------------
    print('\nALL DONE, check {}'.format(str(current_time)))

if __name__ == '__main__':
    main()
