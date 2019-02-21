#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:02:11 2018

@author: eriti
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from scipy.interpolate import interp1d

from global_var import WORKSPACE, DATASPACE, SIM_LENGTH, TLDB

N_IC_STR = list(map(lambda x:str(x), list(range(1, 1+len(TLDB['IC_name'].unique())))))
MAIN_DIRECTION = ['E', 'W']
DIRECTIONS = list(set(['E','W','N','S']) - set(MAIN_DIRECTION))
SUB_DIRECTIONS = list(map(lambda x: x[0] + x[-1], list(product(DIRECTIONS, N_IC_STR))))
ROUTES = list(permutations(MAIN_DIRECTION + SUB_DIRECTIONS,2))
MAIN_ROUTES = list(permutations(MAIN_DIRECTION,2)) #A是有顺序的 C是没有顺序的
SUB_ROUTES = list(set(ROUTES) - set(MAIN_ROUTES)) 

ROUTES_DICT = {}
for each_route in ROUTES:
    form_str = list(map(lambda x: x.replace('E','Ee3').replace('W','Ww1'), each_route))
    begin, end = list(map(lambda x: int(x[-1]), form_str))
    n_edges = end - begin
    real_route = ''
    
    if n_edges == 0:
        edge_list = []
        real_route = 'Edge{}{} Edge{}{}'.format(each_route[0], begin, end, each_route[-1])
    elif n_edges > 0:
        edge_list = list(range(begin, end+1))
        edge_list.insert(0, form_str[0])
        edge_list.append(form_str[1])
    else:
        edge_list = list(range(end, begin+1))[::-1]
        edge_list.insert(0, form_str[0])
        edge_list.append(form_str[1])
    for i in range(len(edge_list)-1):
        try:    
            real_route += 'Edge{}{} '.format(edge_list[i], edge_list[i+1])
        except:
            pass
    real_route = real_route.replace('Ww1','W').replace('Ee3','E')
    ROUTES_DICT[each_route] = real_route


def generateDistribution(array_x, array_y, length = SIM_LENGTH):        
    f1 = interp1d(array_x, array_y, kind = 'quadratic')
    print('The highest expected arrival rate is {} veh/s.'.format(str(array_y.max()/60)))
    xnew = np.linspace(array_x.min(),array_x.max(), int(length))
    ynew = f1(xnew)
    x_smooth = np.array(range(int(length)))
    return x_smooth, ynew

def plotDistribution(x_smooth,ynew,viz_name):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 100)
    plt.plot(x_smooth, ynew)
    plt.xlabel('Time')
    plt.ylabel('Expected arrival rate per minute')
    plt.legend(loc=4)
#    plt.text(0, max(ynew)/10, str(distribution_dict))
    plt.savefig('{}/{}_rou_input.png'.format(DATASPACE, viz_name))
#    plt.show()
    


class homemadeXML():
    def __init__(self, episodes = SIM_LENGTH, interval, binomial, array_y, output_name, verbose = False):
#        self.episodes = episodes # it refers to the duration of simulation
        self.interval = interval # generally it's phase length
        self.binomial = binomial # generally it's a constant number
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
#            veh_edge = 'Edge{}2 Edge{}1'.format(main_direction[0], main_direction[1])
            route = ROUTES_DICT[main_direction]
            # self.flow_distribution[main_direction[0]][main_direction[1]] += 1
        else:
#            veh_edge = 'Edge{}2 Edge{}1'.format(random_direction[0], random_direction[1])
            route = ROUTES_DICT[random_direction]
            # self.flow_distribution[random_direction[0]][random_direction[1]] += 1

        Gericht = '\t<vehicle id="{}" depart="{}"><route edges="{}"/></vehicle>\n'.format(veh_id, veh_depart, route)
        return Gericht
    
    def kochen(self, begin, end, period, binomial):
        depart = begin
        while depart < end:
            if binomial is None:
                # generate with constant spacing
                self.xml_content.append(self.zubereiten(self.veh_number, depart))
                self.veh_number += 1
                depart += period
            else:
                # draw n times from a Bernoulli distribution
                # for an average arrival rate of 1 / period
                prob = 1.0 / period / binomial
                for i in range(binomial):
                    if random.random() < prob:
                        self.xml_content.append(self.zubereiten(self.veh_number, depart))
                        self.veh_number += 1
                    else:
                        pass
                depart += 1
    
    def geniessen(self):
#        proportion = math.floor(len(DISTRIBUTION)/self.pieces)
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


if __name__ == "__main__":
    input_figure_name = 'flow_test'
#    y = np.array([2, 10, 15, 25, 35, 37, 33, 28, 33, 40, 25, 10])
    y = np.array([2, 4, 5, 15, 20, 25, 18, 12, 13, 15, 10, 5])
    x = np.arange(len(y))
    
    X, Y = generateDistribution(x, y) 
    test = homemadeXML(60, 5, Y, input_figure_name, verbose = True)
    test.geniessen()
#    plotDistribution(X, Y, input_figure_name, test.flow_distribution)
    plotDistribution(X, Y, input_figure_name)






