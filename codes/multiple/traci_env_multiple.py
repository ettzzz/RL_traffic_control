import os, sys
SUMO_TOOL = os.path.join(os.environ['SUMO_HOME'], 'tools')
if SUMO_TOOL not in sys.path:
    sys.path.append(SUMO_TOOL)
import traci

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from global_var import trainCmd, TLDB, green_states, SIM_LENGTH

IC_list = TLDB['IC_name'].unique()
                
class myTraci():

    def __init__(self, EPISODES = SIM_LENGTH, sumo_cmd = trainCmd):
        self.conn = traci
        self.conn.start(sumo_cmd)
        self.episodes = EPISODES
        self.intersections = self.conn.trafficlight.getIDList()
        self.tl_params = {'minGrT': 10, 'maxGrT': 40, 'trans':4, 'prolongGrT': 10, 
                        'cycle': 96, 'cycle_test':96,
                        'stGrT':25, 'stGrT_test':25,
                        'lfGrT':15,'lfGrT_test':15}
        self.warmup_time = self.tl_params['cycle'] * 10 + 2
        self.max_prolong = (self.tl_params['maxGrT'] - self.tl_params['minGrT']) / self.tl_params['prolongGrT']
        self.waitingtime = {'Intersection1':[], 'Intersection2':[], 'Intersection3':[],'Sum':[]}
        self.lanes = {'Intersection1':[], 'Intersection2':[], 'Intersection3':[]}
        self.lanes_set = {'Intersection1':[], 'Intersection2':[], 'Intersection3':[]}
        self.cache_vars = {'cp':{'Intersection1':0, 'Intersection2':0, 'Intersection3':0}, 
                           'n_prolong':{'Intersection1':0, 'Intersection2':0, 'Intersection3':0}, 
                           'cache_u':{'Intersection1':np.zeros(40), 'Intersection2':np.zeros(40), 'Intersection3':np.zeros(40)}, 
                           'cache_d1':{'Intersection1':None, 'Intersection2':None, 'Intersection3':None},
                           'cache_d2':{'Intersection1':None, 'Intersection2':None, 'Intersection3':None}}

        self.veh_green = math.floor(self.tl_params['minGrT'] / 2)
        self.max_green = math.floor(self.tl_params['maxGrT'] / 2)
        self.max_dist = math.floor(self.tl_params['stGrT'] / 2) * 7.5


        for each_ic in self.intersections:
            self.lanes[each_ic] = list(self.conn.trafficlight.getControlledLanes(each_ic)) 
            self.lanes[each_ic].sort(key = lambda x:x[-1]) #这个排序方式还得验证一下 暂时看是没问题的
            self.lanes_set[each_ic] = set(self.lanes[each_ic])


    def warmUp(self, category = 'test'): 
        warmup_time_test = self.tl_params['cycle_test'] + 2
        for _ in range(warmup_time_test):
            self.conn.simulationStep()
        for each_ic in self.intersections:
            self.cache_vars['cp'][each_ic] = warmup_time_test + self.tl_params['minGrT']


    def isCheckpoint(self):
        IC_list = []
        for each_ic in self.intersections:
            if self.conn.simulation.getTime() == self.cache_vars['cp'][each_ic]:
                IC_list.append(each_ic)
        return IC_list


    def getTL(self, index = False):
        if index == False:
            return self.conn.trafficlight.getRedYellowGreenState(self.intersection)
        else:
            return self.conn.trafficlight.getPhase(self.intersection)

    # 需哟提前改变一下self.intersection
    def getCurrentLanePair(self):
        a = self.getTL(index = True)//4
        a_oppo = a+2 if a < 1.5 else a-2
        phase = self.lanes[self.intersection][-4:][a]
        phase_opposite = self.lanes[self.intersection][-4:][a_oppo]
        
        if a <= 1: # straight phase
            phase = phase.replace('_1','_0')
            phase_opposite = phase_opposite.replace('_1','_0')
            
        return phase, phase_opposite
    
    def generateEmptyLaneMatrix(self):
        lane_matrix = {}
        for each in self.lanes_set[self.intersection]:
            lane_matrix[each] = np.zeros(math.ceil(self.conn.lane.getLength(each)/7.5))
        return lane_matrix
    
    
    def getCurrentOccasion(self, istest = False):

        idx_bin = ''
        lane1, lane2 = self.getCurrentLanePair()
        spd = self.getLaneMatrix(category = 'speed')
        
        que1, que2 = spd[lane1][:self.veh_green], spd[lane2][:self.veh_green]
        spd1, spd2 = spd[lane1][self.veh_green:self.max_green], spd[lane2][self.veh_green:self.max_green]
        que1n, que2n = np.count_nonzero(que1), np.count_nonzero(que2)
        spd1n = 1 if np.count_nonzero(spd1) == 0 else np.count_nonzero(spd1)
        spd2n = 1 if np.count_nonzero(spd2) == 0 else np.count_nonzero(spd2)

#        idx_que = 0 if que1n <= self.veh_green - 2 and que2n <= self.veh_green - 2 else 1
        idx_que = 0 if que1n == 0 and que2n == 0 else 1
        idx_spd = 1 if np.sum(spd1)/spd1n >= 8.3 and np.sum(spd2)/spd2n >= 8.3 else 0
        idx_bin = str(idx_que) + str(idx_spd) + '{:04b}'.format(self.cache_vars['n_prolong'][self.intersection])
#        print(idx_bin)        
        idx_dec = int(idx_bin, 2)  # 0 - 255
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
        lane_list = self.lanes_set[self.intersection] - set(self.getCurrentLanePair()) if category == 'red' else self.lanes_set[self.intersection]
        for each_lane in lane_list:
            utility += np.sum(wt[each_lane][:self.max_green])
        return round(utility,1)

    def getLaneMatrix(self, category): # category: isCar, wt, speed, accel, 
        cache_matrix = self.generateEmptyLaneMatrix()
        for each_lane in self.lanes_set[self.intersection]:
           veh_list = self.conn.lane.getLastStepVehicleIDs(each_lane)
           for each_veh in veh_list:
                grid_pos = math.floor((self.conn.lane.getLength(each_lane) - self.conn.vehicle.getLanePosition(each_veh))/7.5)
                if category == 'wt':
                    each_wt = self.conn.vehicle.getAccumulatedWaitingTime(each_veh)
                    cache_matrix[each_lane][grid_pos] = each_wt if each_wt != 0 else 0.0001
                    # check edge.getWaitingTime()
                elif category == 'speed':
                    each_speed = self.conn.vehicle.getSpeed(each_veh)
                    cache_matrix[each_lane][grid_pos] = round(each_speed,2) if each_speed <= 13.9 else 13.9
#                    cache_matrix[each_lane][grid_pos] = round(each_speed,2) if each_speed <= 0 else 0.0001
                else:
                    print('Bug from getLaneMatrix(): Wrong category keyword')
                    break
                
        return cache_matrix


    def prolongTL(self, action):
        if action == 1 and self.cache_vars['n_prolong'][self.intersection] <= self.max_prolong:
            self.conn.trafficlight.setPhaseDuration(self.intersection, self.tl_params['prolongGrT'])
            self.cache_vars['n_prolong'][self.intersection] += 1
            self.cache_vars['cp'][self.intersection] += self.tl_params['prolongGrT']
        else:
            self.cache_vars['n_prolong'][self.intersection] = 0
            self.cache_vars['cp'][self.intersection] += self.tl_params['minGrT'] + self.tl_params['trans']
            self.conn.trafficlight.setPhaseDuration(self.intersection, 0)
        self.cache_vars['cache_u'][self.intersection] = np.zeros(20)
        

    def calWaitingTime(self):
        for each_ic in self.intersections:
            self.intersection = each_ic
            self.waitingtime[each_ic].append(self.getNetworkDelay())

        sum_wt = 0
        for each_ic in self.intersections:
            sum_wt += self.waitingtime[each_ic][-1]
        self.waitingtime['Sum'].append(sum_wt)

    def plotScatter(self, y, sim_path):
        x = np.arange(1, len(y) + 1)
        plt.ioff()
        plt.figure(figsize = (12,6),dpi = 100)
        plt.scatter(x, np.array(y))
        plt.plot(x, np.array(y))
        plt.xlabel('Episodes')
        plt.ylabel('Avg. waiting time')
        plt.savefig('{}/WT_scatter.png'.format(sim_path))
#        plt.show()
#        print(y)

    def getReward(self):
        return  round(np.sum(self.cache_vars['cache_u'][self.intersection])
                      - self.cache_vars['cache_d2'][self.intersection]
                      + self.cache_vars['cache_d1'][self.intersection] , 1) 
    

    def getUtility(self, step):
        
        for each_ic in self.intersections:
            self.intersection = each_ic
            if self.getTL() in green_states:
                t0 = int(self.cache_vars['cp'][self.intersection] - step)
                if t0 == self.tl_params['minGrT'] - 1:
                    self.cache_vars['cache_d1'][self.intersection] = self.getNetworkDelay('red')
                if t0 == 0:
                    self.cache_vars['cache_d2'][self.intersection] = self.getNetworkDelay('red')
                
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
#                print(step,self.intersection, self.cache_vars['cp'][self.intersection], t0, round(td1+td2,1),  self.getNetworkDelay())
                self.cache_vars['cache_u'][self.intersection][t0] = round(td1 + td2, 1)
            else:
                pass
        
#from RL_brain import QLearningTable
#tempCmd = ["{}/bin/sumo-gui".format(os.environ['SUMO_HOME']), "-c", "{}/temp.sumocfg".format(WORK_SPACE)]
#RL = QLearningTable(list(range(len(tl_states))))
#feed_path = '/Users/eriti/Desktop/frompps/02160331/qtable.csv'
#RL.feed_qtable(feed_path)
#
#env = myTraci(EPISODES = 7200, sumo_cmd = tempCmd) # need modification when using in multiple scenario
#env.warmUp()
#step = env.conn.simulation.getTime()
#
#while step < 1600:
#    env.getUtility(step) # right after warmup
#    ic_list = env.isCheckpoint()
#    if len(ic_list) != 0:
#        for each_ic in ic_list:
#            env.intersection = each_ic
#            o, s = env.getCurrentOccasion()
#            a = RL.choose_action(s, o) # get action
#            env.prolongTL(a)
##            break
##    if step % VERIFY_INTERVAL == 0:
##        env.calWaitingTime()
#
#    step += 1
#    env.conn.simulationStep()
#env.conn.close()
#sys.stdout.flush()
##    
