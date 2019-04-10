#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:46:14 2019

@author: eriti
"""
from RL_brain import QLearningTable
from train_agent import generatePath, writeLog, trainAgent
from global_var import green_states, current_time, WORKSPACE
from verify_single import testAgent, plotTestResult
from flow_input_single import pushAgent


def main():
    # --------------preparation--------------------
    rst_path, sim_path = generatePath(current_time) # Create a new folder for the experiment
    RL = QLearningTable(list(range(len(green_states)))) # Initialize the Q-learning framework
    feed_path = '{}/results/{}/qtable.csv'.format(WORKSPACE, 'p5i3g0')
    RL.feedQTable(feed_path) # This could be helpful when inheriting from previous trained agent
    # ---------------training--------------------
    trainAgent(RL, rst_path, sim_path)
    # --------------testing--------------------
    RL.epsilon = 1 # Epsilon-greedy no longer selects random actions
    fixed, rl, actuated = testAgent('fixed', RL), testAgent('rl', RL), testAgent('actuated', RL)
    plotTestResult(rl, fixed, actuated, sim_path)
    flow_scenarios = ['-50%', '-25%',  '0%', '+25%', '+50%']
    pushAgent(flow_scenarios, sim_path, RL) # Explore the limit of the trained agent
    # --------------results----------------------
    RL.saveQTable('{}/qtable.csv'.format(sim_path))
    RL.plotCumulativeReward(sim_path) # Plot the cumulative reward
    RL_params = {'lr':RL.alpha, 'gamma':RL.gamma, 'e_max':RL.e_greedy_max, 'e_inc':RL.e_greedy_increment}
    writeLog(RL_params, rst_path, sim_path, clean = True) # Record some basic information of the experiment
    # --------------end--------------------
    print('\nALL DONE, check {}'.format(str(current_time)))

if __name__ == '__main__':
    main()
