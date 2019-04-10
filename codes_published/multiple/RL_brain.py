#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:40:04 2019

@author: eriti
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from global_var import TLDB
#np.random.seed(1)

class QLearningTable:
    def __init__(self, actions, learning_rate=0.001, reward_decay=0, e_greedy_max=0.95, e_greedy_increment = None):
        self.actions = actions  
        self.alpha = learning_rate
        self.gamma = reward_decay

        self.e_greedy_max = e_greedy_max
        self.e_greedy_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.e_greedy_max

        self.learnlog = pd.DataFrame(columns = ['epi', 'step', 's', 'o', 'a', 'r'])

        self.batch_size = 1000
        self.learn_size = 500
        self.batch_memory = np.zeros((self.batch_size, 4), dtype = np.int16)
        self.cache_vars = {'n_batch':0, 'n_learn':0}
    
        self.q_table = pd.DataFrame(columns = self.actions, dtype=np.float64)
        for i in range(2**6):
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index = self.q_table.columns, name = '0'))
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index = self.q_table.columns, name = '1'))
        self.q_table.reset_index(inplace = True)
            
        
    def chooseAction(self, observation, occasion):
        # action selection
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[occasion*2:occasion*2+1, observation] # choose a row    
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)%2 # choose a column
            # some actions may have the same value, randomly choose on in these actions
        else:
            # choose random action
            action = np.random.choice(np.array([0,1]))
        return int(action)
    
    
    def learn(self):
        learn_list = np.random.randint(0, self.batch_size, self.learn_size)
        for i in learn_list:
            if random.random() >= self.learn_size/self.batch_size:
                n_qt, s, a, r = self.batch_memory[i]
                q_predict = self.q_table.loc[n_qt * 2 + a, s]
                q_target = r + self.gamma * self.q_table.iloc[n_qt*2:n_qt*2 + 2].loc[:, s,].max()
                self.q_table.loc[n_qt*2+a,s] += round(self.alpha * (q_target - q_predict),3)

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy_max else self.e_greedy_max
        

    def writeLearnLog(self, epi, current_step, s, o, a, r): # record everytime agent learns
        s0 = TLDB[(TLDB['Q_index'] == s)]['Alias'].values[0].split(':')[0]
        a0 = 'prolong' if a == 1 else 'pass'
        o0 = 'bin+' + '{:06b}'.format(o)[:2] + '+' + str(int('{:06b}'.format(o)[2:],2))
        log = [epi, current_step, s0, o0, a0, r]
        self.learnlog.loc[self.learnlog.shape[0]+1] = log

    def writeMemoryRecord(self, o, s, a, r): # everytime when it's green end
        if self.cache_vars['n_batch'] < self.batch_size:
            record = [o, s, a, r]
            self.batch_memory[self.cache_vars['n_batch']] = record
            self.cache_vars['n_batch'] += 1
        else:
            self.learn()
            self.cache_vars['n_batch'] = 0

    def plotCumulativeReward(self, path):
        cr = []
        for i in range(self.learnlog['epi'].unique()[-1]):
            reward = sum(self.learnlog[(self.learnlog['epi'] == i)]['r'])
            cr.append(reward)

        plt.ioff()
        plt.figure(figsize = (12,6),dpi = 400)
        plt.plot(cr)
        plt.xlabel('Episodes', fontsize = 12)
        plt.ylabel('Cumulative reward', fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.savefig(path + 'cumulative_reward.png')
        # plt.show()
        plt.close('all')


    def saveQTable(self, path):
        try:
            reference = TLDB[(TLDB['isGreen'] == 1)][['Q_index','Alias']].set_index('Q_index').T.to_dict('record')[0]
            cache_table = self.q_table
            cache_table.columns = cache_table.columns.map(reference)
            cache_table.to_csv(path, sep = ';', index = False)
        except:
            print('Bug from saveQTable(): Your reference is illegal, need debug. from saveQTable')        

    def feedQTable(self, path): 
        try:
            reference = TLDB[(TLDB['isGreen'] == 1)][['Q_index','Alias']].set_index('Alias').T.to_dict('record')[0]
            cache_table = pd.read_csv(path, index_col = 0, sep = ';')
            cache_table.columns = cache_table.columns.map(reference)
            self.q_table = cache_table
            self.q_table.reset_index(inplace = True)
        except:
            print('Bug from feedQTable(): Your reference is illegal, need debug. from feedQTable')   


