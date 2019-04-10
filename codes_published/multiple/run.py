#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:46:14 2019

@author: eriti
"""
import os

from global_var import green_states, WORKSPACE
from RL_brain import QLearningTable
from verify_multiple import testAgent, plotTestResult


def getLastExperiment(which_one_do_you_like):
    if which_one_do_you_like == None:
        rst_path = WORKSPACE + '/results/'
        results_list = os.listdir(rst_path)
        return results_list[-1]
    else:
        return which_one_do_you_like

def main():
    trained_number = getLastExperiment('p5i3g0')
    RL = QLearningTable(list(range(len(green_states))))
    trained_path = '{}/results/{}/'.format(WORKSPACE, trained_number)
    qtable_path = trained_path + 'qtable.csv'
    RL.feedQTable(qtable_path)
    RL.epsilon = 1
    fixed,rl,actuated = testAgent('fixed', RL), testAgent('rl', RL), testAgent('actuated', RL)
    plotTestResult(rl, fixed, actuated, trained_path)

if __name__ == '__main__':
    main()
    

