#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:46:14 2019

@author: eriti
"""
import os

from global_var import green_states, WORKSPACE
from RL_brain import QLearningTable
from verify_multiple import test_agent, test_plot


def get_last_number():
    rst_path = WORKSPACE + '/results/'
    results_list = os.listdir(rst_path)
    return results_list[-1]

if __name__ == '__main__':
    # trained_number = '02180840'
    trained_number = get_last_number()
    RL = QLearningTable(list(range(len(green_states))))
    trained_path = '{}/results/{}/'.format(WORKSPACE, trained_number)
    qtable_path = trained_path + 'qtable.csv'
    RL.feed_qtable(qtable_path)
    fixed,rl,actuated = test_agent('fixed', RL), test_agent('rl', RL), test_agent('actuated', RL)
    test_plot(rl, fixed, actuated, trained_path)
    

