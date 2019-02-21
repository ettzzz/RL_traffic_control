#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:40:04 2019

@author: eriti
"""


'''
RL_brain.py 适配不改变tl顺序的版本
'''


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from global_var import TLDB

#np.random.seed(1)

class QLearningTable:
    def __init__(self, actions, learning_rate=0.001, reward_decay=0, e_greedy_max=0.95, e_greedy_increment = None):
        self.actions = actions  
        self.lr = learning_rate
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
            
        
    def choose_action(self, observation, occasion):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[occasion*2:occasion*2+1, observation] # choose a row
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)%2 # choose a column
        else:
            # choose random action
            action = np.random.choice(np.array([0,1]))
        return int(action)


    
    def learn(self):
        learn_list = np.random.randint(0, self.batch_size, self.learn_size)
        for i in learn_list:
            if random.random() >= self.learn_size/self.batch_size:
                n_qt, s, a, r = self.batch_memory[i]
                #q_predict = self.q_table.iloc[n_qt * 2:n_qt * 2 + 2].iloc[a, s] #会偶尔出现str的情况？？？？
                q_predict = self.q_table.loc[n_qt * 2 + a, s]
                q_target = r + self.gamma * self.q_table.iloc[n_qt*2:n_qt*2 + 2].loc[:, s,].max()
                self.q_table.loc[n_qt*2+a,s] += round(self.lr * (q_target - q_predict),3)

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy_max else self.e_greedy_max
        

    def write_log(self, epi, current_step, s, o, a, r): # record everytime agent learns
        s0 = TLDB[(TLDB['Q_index'] == s)]['Alias'].values[0].split(':')[0]
        a0 = 'prolong' if a == 1 else 'pass'
        o0 = 'bin+' + '{:06b}'.format(o)[:2] + '+' + str(int('{:06b}'.format(o)[2:],2))
        log = [epi, current_step, s0, o0, a0, r]
        self.learnlog.loc[self.learnlog.shape[0]+1] = log

    def write_memory_record(self, o, s, a, r): # everytime when it's green end
        if self.cache_vars['n_batch'] < self.batch_size:
            record = [o, s, a, r]
            self.batch_memory[self.cache_vars['n_batch']] = record
            self.cache_vars['n_batch'] += 1
        else:
            self.learn()
            self.cache_vars['n_batch'] = 0

    def plot_cumulative_reward(self, path):
        cr = []
        for i in range(self.learnlog['epi'].unique()[-1]):
            reward = sum(self.learnlog[(self.learnlog['epi'] == i)]['r'])
            cr.append(reward)

        plt.ioff()
        plt.figure(figsize = (12,6),dpi = 120)
        plt.plot(cr)
        plt.xlabel('Simulation runs')
        plt.ylabel('Cumulative reward')
        # plt.title(IC_name)
        plt.savefig(path + 'cumulative_reward.png')
        # plt.show()
        plt.close('all')


    def save_qtable(self, path):
        try:
            reference = TLDB[(TLDB['isGreen'] == 1)][['Q_index','Alias']].set_index('Q_index').T.to_dict('record')[0]
            cache_table = self.q_table # 这里不应该啊 cache_table和self。q_table不应该是两个不同的变量嘛？
            cache_table.columns = cache_table.columns.map(reference)
            cache_table.to_csv(path, sep = ';', index = False)
        except:
            print('Bug from save_qtable(): Your reference is illegal, need debug. from save_qtable')        

    def feed_qtable(self, path): 
        try:
            reference = TLDB[(TLDB['isGreen'] == 1)][['Q_index','Alias']].set_index('Alias').T.to_dict('record')[0]
            cache_table = pd.read_csv(path, index_col = 0, sep = ';')
            cache_table.columns = cache_table.columns.map(reference)
            self.q_table = cache_table
            self.q_table.reset_index(inplace = True)
        except:
            print('Bug from feed_qtable(): Your reference is illegal, need debug. from feed_qtable')   

#
#if __name__ == '__main__':
#    from RL_global_var import tl_states
#    RL = QLearningTable(list(range(len(tl_states))))
#    feed = '/Users/eriti/Desktop/netedit/single/results/02180840_4idxs/'
#    RL.learning_log = pd.read_csv(feed + 'learnlog.csv', sep=';')
#    RL.plot_cumulative_reward(feed)
