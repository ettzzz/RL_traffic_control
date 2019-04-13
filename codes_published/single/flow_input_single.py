#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:02:11 2018

@author: eriti
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.interpolate import interp1d

from global_var import WORKSPACE, DATASPACE, SIM_DURATION
from verify_single import testAgent

ROUTES = list(permutations(['N','E','W','S'],2)) # Permutation with sequence
MAIN_ROUTES = list(permutations(['E','W'],2))
SUB_ROUTES = list(set(ROUTES) - set(MAIN_ROUTES)) 
#ROUTES = list(combinations(['N','E','W','S'],2)) # Combination without sequence

def generateDistribution(array_x, array_y, length = SIM_DURATION):        
    f1 = interp1d(array_x, array_y, kind = 'quadratic')
    xnew = np.linspace(array_x.min(),array_x.max(), int(length))
    ynew = f1(xnew)
    x_smooth = np.array(range(int(length)))
    return x_smooth, ynew

def plotDistribution(x_smooth, ynew, viz_name, distribution_dict):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 400)
    plt.plot(x_smooth, ynew/60)
    plt.xlabel('Time', fontsize = 12)
    plt.ylabel('Expected arrival rate per second', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.text(0, max(ynew)/10, str(distribution_dict))
    plt.savefig('{}/{}_input.png'.format(DATASPACE, viz_name))
    # plt.show()
    plt.close('all')
    

class homemadeXML():
    def __init__(self, interval, binomial, array_y, output_name, episodes = SIM_DURATION, verbose = False):
        self.interval = interval # how often does the flow generate, 60 seconds as default
        self.binomial = binomial # n in binomial distribution
        self.distribution = array_y # note this is a np.array form
        self.pieces = int(episodes/interval)
        
        self.veh_number = 0
        self.xml_content = []
        self.head = '<?xml version="1.0" encoding="UTF-8"?>\n<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        self.tail = '</routes>\n'
        self.output_name = output_name
        self.verbose = verbose
        self.flow_distribution = {'N':{'E':0,'S':0,'W':0},
                                  'E':{'S':0,'W':0,'N':0},
                                  'W':{'N':0,'E':0,'S':0},
                                  'S':{'W':0,'N':0,'E':0}}
            
    def zubereiten(self, veh_id, veh_depart):
        random_direction = random.choice(SUB_ROUTES)
        main_direction = random.choice(MAIN_ROUTES)

        prob = 0.2
        if random.random() < prob:
            veh_edge = 'Edge{}2 Edge{}1'.format(main_direction[0], main_direction[1])
            self.flow_distribution[main_direction[0]][main_direction[1]] += 1
        else:
            veh_edge = 'Edge{}2 Edge{}1'.format(random_direction[0], random_direction[1])
            self.flow_distribution[random_direction[0]][random_direction[1]] += 1
            
        Gericht = '\t<vehicle id="{}" depart="{}"><route edges="{}"/></vehicle>\n'.format(veh_id, veh_depart, veh_edge)
        return Gericht
    
    def kochen(self, begin, end, period, binomial):
        depart = begin
        while depart < end:
            if binomial is None:
                # generate vehicle flow with constant interval
                self.xml_content.append(self.zubereiten(self.veh_number, depart))
                self.veh_number += 1
                depart += period
            else:
                # draw n times from a Bernoulli distribution for an average arrival rate of 1 / period
                prob = 1.0 / period / binomial
                for i in range(binomial):
                    if random.random() < prob:
                        self.xml_content.append(self.zubereiten(self.veh_number, depart))
                        self.veh_number += 1
                    else:
                        pass
                depart += 1
    
    def geniessen(self):
        for piece in range(self.pieces):
            b = self.interval * piece
            e = b + self.interval
            p = 60/self.distribution[b]
            bi = self.binomial
            self.kochen(b,e,p,bi)
            
        with open ('{}/{}.rou.xml'.format(DATASPACE, self.output_name), 'w') as output:
            output.write(self.head)
            for each_xml in self.xml_content:
                output.write(each_xml)
            output.write(self.tail)
            
        if self.verbose == True:
            print('Kochen fertig. Bitte sehen Sie das Folder an. Es gibt {} Verkehrs während die Simulation.'.format(str(self.veh_number)))

    
def pushAgent(test_scenarios, path, RL):
    def transfer(sce_str):
        if sce_str[0] == '-':
            return int(sce_str.replace('%',''))/100
        else:
            return int(sce_str.replace('%','').replace('+',''))/100
        
    input_figure_name = 'flow_test'
    y = np.array([8, 10, 12, 13, 15, 15, 14, 13, 14, 16, 12, 7])
    x = np.arange(len(y))
    test_results = [[] for _ in range(len(test_scenarios))]
    for each_sce in test_scenarios:
        y_sce = y * (1 + transfer(each_sce))
        
        X, Y = generateDistribution(x, y_sce) 
        flow = homemadeXML(60, 5, Y, input_figure_name, verbose = False)
        flow.geniessen()
        test_results[0].append(np.mean(testAgent('fixed', RL)))
        test_results[1].append(np.mean(testAgent('rl', RL)))
        test_results[2].append(np.mean(testAgent('actuated', RL)))
    
    X, Y = generateDistribution(x, y) 
    reset = homemadeXML(60, 5, Y, input_figure_name, verbose = False)
    reset.geniessen()
    
    index = np.arange(len(test_scenarios))  # the x locations for the groups
    width = 0.1  # the width of the histogram
    
    fig, ax = plt.subplots(figsize = (12,6),dpi = 400)
    ax.bar(index - width, test_results[0], width, color = '#006ED4', label='Fixed')
    ax.bar(index, test_results[1], width, color = '#FF341F', label='RL agent')
    ax.bar(index + width, test_results[2], width, color = '#30AA52', label='Adaptive')
    
    ax.set_ylabel('Avg. Waiting Time')
    ax.set_xticks(index)
    ax.set_xticklabels(test_scenarios)
    ax.legend()
    plt.savefig(path + 'scenarios_bar.png')
    plt.close('all')
    

if __name__ == "__main__":
    input_figure_name = 'flow_test' # change the name into flow_train when necessary
    y = np.array([8, 10, 12, 13, 15, 15, 14, 13, 14, 16, 12, 7]) 
    # User-defined traffic flow tendency, Also known as temporal distribution
    x = np.arange(len(y))
    X, Y = generateDistribution(x, y) 
    flow = homemadeXML(60, 5, Y, input_figure_name, verbose = True)
    flow.geniessen()
    plotDistribution(X, Y, input_figure_name, flow.flow_distribution)








