import os, sys
SUMO_TOOL = os.path.join(os.environ['SUMO_HOME'], 'tools')
if SUMO_TOOL not in sys.path:
    sys.path.append(SUMO_TOOL)
import traci

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from global_var import trainCmd, TLDB, green_states, WORKSPACE, SIM_LENGTH

IC_list = TLDB['IC_name'].unique()
                
class myTraci():

    def __init__(self, EPISODES = SIM_LENGTH, AGENT='Intersection0', sumo_cmd = trainCmd):
        self.conn = traci
        self.conn.start(sumo_cmd)
        self.episodes = EPISODES
        self.intersection = AGENT
        self.tl_params = {'minGrT': 10, 'maxGrT': 40, 'trans':4, 'prolongGrT': 10, 
                        'cycle': 96, 'cycle_test':96,
                        'stGrT':25, 'stGrT_test':25,
                        'lfGrT':15,'lfGrT_test':15,}
        self.warmup_time = self.tl_params['cycle'] * 10 + 2 # 1s for pre-yellow, 1s for start of green
        self.max_prolong = (self.tl_params['maxGrT'] - self.tl_params['minGrT']) / self.tl_params['prolongGrT']
        self.waitingtime = []
        
        self.cache_vars = {'n_prolong':0, 'cp':0, 'cache_u':np.zeros(20), 'cache_d1':None, 'cache_d2':None}
        self.veh_green = math.floor(self.tl_params['minGrT'] / 2)
        self.max_green = math.floor(self.tl_params['maxGrT'] / 2)
        self.max_dist = math.floor(self.tl_params['stGrT'] / 2) * 7.5

    def warmUp(self, category = None):
        self.lanes = list(self.conn.trafficlight.getControlledLanes(self.intersection))
        self.lanes.sort(key = lambda x:x[-1])
        self.lanes_set = set(self.lanes)

        if category == 'test':
            warmup_time_test = self.tl_params['cycle_test'] + 2
            for _ in range(warmup_time_test):
                self.conn.simulationStep()
            self.cache_vars['cp'] = warmup_time_test + self.tl_params['minGrT']   
        else:
            for _ in range(self.warmup_time):
                self.conn.simulationStep()
            self.cache_vars['cp'] = self.warmup_time + self.tl_params['minGrT']

            
    def isCheckpoint(self):
        if self.conn.simulation.getTime() == self.cache_vars['cp']:
            
            return True
        else:
            return False
        

    def getTL(self, index = False):
        if index == False:
            return self.conn.trafficlight.getRedYellowGreenState(self.intersection)
        else:
            return self.conn.trafficlight.getPhase(self.intersection)
    
    
    def generateEmptyLaneMatrix(self):
        lane_matrix = {}
        for each in self.lanes_set:
            lane_matrix[each] = np.zeros(math.ceil(self.conn.lane.getLength(each)/7.5))
        return lane_matrix
    
    
    def getCurrentLanePair(self):
        a = self.getTL(index = True)//4
        phase = self.lanes[-4:][a]
        if a <= 1: # straight phase
            phase = phase.replace('1','0')
            phase_opposite = phase.replace('N2','S2').replace('E2','W2')
        else:
            phase_opposite = phase.replace('S2','N2').replace('W2','E2')
        return phase, phase_opposite
    
    
    def getCurrentOccasion(self, istest = False):
        idx_bin = ''
        lane1, lane2 = self.getCurrentLanePair()
        spd = self.getLaneMatrix(category = 'speed')
        
        que1, que2 = spd[lane1][:self.veh_green], spd[lane2][:self.veh_green]
        spd1, spd2 = spd[lane1][self.veh_green:self.max_green], spd[lane2][self.veh_green:self.max_green]
        que1n, que2n = np.count_nonzero(que1), np.count_nonzero(que2)
        spd1n = 1 if np.count_nonzero(spd1) == 0 else np.count_nonzero(spd1)
        spd2n = 1 if np.count_nonzero(spd2) == 0 else np.count_nonzero(spd2)

        idx_que = 0 if que1n <= self.veh_green - 2 and que2n <= self.veh_green - 2 else 1
#        idx_que = 0 if que1n == 0 and que2n == 0 else 1
        idx_spd = 0 if np.sum(spd1)/spd1n >= 8.3 and np.sum(spd2)/spd2n >= 8.3 else 1
        idx_bin = str(idx_que) + str(idx_spd) + '{:04b}'.format(self.cache_vars['n_prolong'])
#        print(idx_bin)        
        idx_dec = int(idx_bin, 2)
        phase = green_states.index(self.getTL())
        
        return idx_dec, phase


    def getCurrentLaneInfo(self, category = 'Lane'):
        cs = self.getTL() # cs = current state
        if 'G' not in cs:
            print('Bug from getCurrentLaneInfo(): Phase error, need debug', cs)
        else:
            lane1, lane2 = self.getCurrentLanePair()
            if category == 'Lane':
                return lane1, lane2
            elif category == 'Edge':
                return self.conn.lane.getEdgeID(lane1), self.conn.lane.getEdgeID(lane2)
            elif category == 'Length':
                return self.conn.lane.getLength(lane1), self.conn.lane.getLength(lane2)
            elif category == 'ID':
                return self.conn.lane.getLastStepVehicleIDs(lane1)[::-1], self.conn.lane.getLastStepVehicleIDs(lane2)[::-1]
            else:
                print('Bug from getCurrentLaneInfo(): Wrong category keyword')


    def getNetworkDelay(self, category = 'all'):
        utility = 0
        wt = self.getLaneMatrix(category = 'wt') 
        lane_list = self.lanes_set - set(self.getCurrentLanePair()) if category == 'red' else self.lanes_set
        for each_lane in lane_list:
            utility += np.sum(wt[each_lane][:self.max_green])
        return round(utility,1)


    def getLaneMatrix(self, category): # category: isCar, wt, speed, accel, 
        cache_matrix = self.generateEmptyLaneMatrix()
        for each_lane in self.lanes_set:
           veh_list = self.conn.lane.getLastStepVehicleIDs(each_lane)
           for each_veh in veh_list:
                grid_pos = math.floor((self.conn.lane.getLength(each_lane) - self.conn.vehicle.getLanePosition(each_veh))/7.5)
                if category == 'wt':
                    each_wt = self.conn.vehicle.getAccumulatedWaitingTime(each_veh)
                    cache_matrix[each_lane][grid_pos] = each_wt if each_wt != 0 else 0.0001
                elif category == 'speed':
                    each_speed = self.conn.vehicle.getSpeed(each_veh)
                    cache_matrix[each_lane][grid_pos] = round(each_speed,2) if each_speed <= 13.9 else 13.9
                else:
                    print('Bug from getLaneMatrix(): Wrong category keyword')
                    break
                
        return cache_matrix


    def getUtility(self, step):
        s = self.getTL()
        if s in green_states:
            t0 = int(self.cache_vars['cp'] - step)
            if t0 == self.tl_params['minGrT'] - 1:
                self.cache_vars['cache_d1'] = self.getNetworkDelay('red')
            if t0 == 0:
                self.cache_vars['cache_d2'] = self.getNetworkDelay('red')
            
            edge1, edge2 = self.getCurrentLaneInfo('Edge')
            length1, length2 = self.getCurrentLaneInfo('Length')
            id1, id2 = self.getCurrentLaneInfo('ID') # 要看这个id从前到后还是从后到前
            td1, td2 = 0,0
            
            for each_id1 in id1:
                dist = self.conn.vehicle.getDrivingDistance(each_id1, edge1, length1)
                if dist <= self.max_dist:
                    d = t0 - dist/13.9 if self.conn.vehicle.getSpeed(each_id1) < 13.9 else 0
                    td1 += d
                else:
                    break
                
            for each_id2 in id2:
                dist = self.conn.vehicle.getDrivingDistance(each_id2, edge2, length2)
                if dist <= self.max_dist:
                    d = t0 - dist/13.9 if self.conn.vehicle.getSpeed(each_id2) < 13.9 else 0
                    td2 += d
                else:
                    break
#            print(step,self.getTL(), t0, round(td1+td2,1),  self.getNetworkDelay())
            self.cache_vars['cache_u'][t0] = round(td1 + td2, 1)
        else:
            pass
        
        
    def prolongTL(self, action):
        if action == 1 and self.cache_vars['n_prolong'] < self.max_prolong:
            self.conn.trafficlight.setPhaseDuration(self.intersection, self.tl_params['prolongGrT'])
            self.cache_vars['n_prolong'] += 1
            self.cache_vars['cp'] += self.tl_params['prolongGrT']
        else:
            self.cache_vars['n_prolong'] = 0
            self.cache_vars['cp'] += self.tl_params['minGrT'] + self.tl_params['trans']
            self.conn.trafficlight.setPhaseDuration(self.intersection, 0)
        self.cache_vars['cache_u'] = np.zeros(20)
        
        
    def calWaitingTime(self):
        self.waitingtime.append(self.getNetworkDelay())
        
        
    def getReward(self):
        return  round(np.sum(self.cache_vars['cache_u'])
                      - (self.cache_vars['cache_d2'] - self.cache_vars['cache_d1']), 1)


    def plotScatter(self, y, sim_path):
        x = np.arange(1, len(y) + 1)
        plt.ioff()
        plt.figure(figsize = (12,6),dpi = 100)
        plt.scatter(x, np.array(y))
        plt.plot(x, np.array(y))
        plt.xlabel('Episodes')
        plt.ylabel('Avg. waiting time')
        plt.savefig('{}/waitingtime_scatter.png'.format(sim_path))
#        plt.show()
#        print(y)

   

            
    
    
#
#from RL_brain import QLearningTable
#tempCmd = ["{}/bin/sumo-gui".format(os.environ['SUMO_HOME']), "-c", "{}/temp.sumocfg".format(WORK_SPACE)]
#
#RL = QLearningTable(list(range(len(tl_states))))
#env = myTraci(EPISODES = 7200, sumo_cmd = tempCmd) # need modification when using in multiple scenario
#env.warmUp()
#step = env.conn.simulation.getTime()
##
##while step < env.episodes:
#while step < 3200:
#    env.getUtility(step) # right after warmup
#    if env.isCheckpoint():
#        r = env.getReward()
#        o, s = env.getCurrentOccasion()
#        a = RL.choose_action(s, o) # get action
#        
#        env.prolongTL(a)
#        
##        print(o,s,a,r)
##        RL.write_memory_record(o, s, a, r)
##        break
#    else:
#        pass
#    step += 1
#    env.conn.simulationStep()
##    
#env.conn.close()
#sys.stdout.flush()


