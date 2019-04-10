
import sys
import numpy as np
import matplotlib.pyplot as plt

from traci_env_multiple import myTraci
from global_var import VERIFY_INTERVAL, testCmd, actCmd

def plotTestResult(rl, fixed, actuated, simulation_path):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 400)
    nrow, ncol = 2, 2
    for i in range(nrow*ncol):
        IC_name = 'Intersection' + str(i+1) if i + 1 <= 3 else 'Sum'
        plt.subplot(nrow, ncol, i+1)
        plt.plot(np.arange(len(rl[IC_name]))*VERIFY_INTERVAL, np.array(rl[IC_name]), color = '#FF341F', label = 'RL')
        plt.plot(np.arange(len(fixed[IC_name]))*VERIFY_INTERVAL, np.array(fixed[IC_name]), color = '#006ED4', label= 'Fixed')
        plt.plot(np.arange(len(actuated[IC_name]))*VERIFY_INTERVAL, np.array(actuated[IC_name]),  color = '#30AA52', label= 'Adaptive')
        plt.xlabel('Time', fontsize = 12)
        plt.ylabel('Total waiting time', fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.legend(loc=4)
        plt.title(IC_name)
    
    plt.tight_layout()
    plt.savefig(simulation_path + 'verify_multiple.png')
    # plt.show()
    plt.close('all')


def testAgent(sim_type, RL):
    if sim_type =='actuated':
        env = myTraci(sumo_cmd = actCmd)
    else:
        env = myTraci(sumo_cmd = testCmd)
        
    env.warmUp(category = 'test')
    step = env.conn.simulation.getTime()
    print('Simulating for {}.'.format(sim_type))
    
    while step < env.duration:
        if sim_type == 'rl':             
            env.getUnutilizedGreen(step)
            ic_list = env.isCheckpoint()
            if len(ic_list) != 0:
                for each_ic in ic_list:
                    env.intersection = each_ic
                    o, s = env.getCurrentOccasion()
                    a = RL.chooseAction(s, o)
                    env.prolongTL(a)
                
        if step % VERIFY_INTERVAL == 0:
            env.calWaitingTime()

        step += 1
        env.conn.simulationStep()
    env.conn.close()
    sys.stdout.flush()
    
    print(round(np.mean(env.waitingtime['Sum']), 2), len(env.waitingtime['Sum']))
    return env.waitingtime







