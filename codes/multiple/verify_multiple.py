
import sys
import numpy as np
import matplotlib.pyplot as plt

from traci_env_multiple import myTraci
from global_var import VERIFY_INTERVAL, testCmd, actCmd

def test_plot(rl, fixed, actuated, simulation_path):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 120)
    nrow, ncol = 2, 2
    for i in range(nrow*ncol):
        IC_name = 'Intersection' + str(i+1) if i + 1 <= 3 else 'Sum'
        plt.subplot(nrow, ncol, i+1)
        plt.plot(np.arange(len(rl[IC_name]))*VERIFY_INTERVAL, np.array(rl[IC_name]), color = 'red', label = 'RL')
        plt.plot(np.arange(len(fixed[IC_name]))*VERIFY_INTERVAL, np.array(fixed[IC_name]), label= 'Fixed')
        plt.plot(np.arange(len(actuated[IC_name]))*VERIFY_INTERVAL, np.array(actuated[IC_name]), label= 'Actuated')
        plt.xlabel('Time')
        plt.ylabel('Total delay time')
        plt.legend(loc=4)
        plt.title(IC_name)
    
    plt.tight_layout()
    plt.savefig(simulation_path + 'verify_multiple.png')
#    plt.show()
    plt.close('all')


def test_agent(sim_type, RL):
    if sim_type =='actuated':
        env = myTraci(sumo_cmd = actCmd)
    else:
        env = myTraci(sumo_cmd = testCmd)
        
    env.warmUp(category = 'test')# because in test cases we use different TL program and cycle time is different
    step = env.conn.simulation.getTime()
    print('Simulating for {}.'.format(sim_type))
    
    while step < env.episodes:
        if sim_type == 'rl':             
            env.getUtility(step) # right after warmup
            ic_list = env.isCheckpoint()
            if len(ic_list) != 0:
                for each_ic in ic_list:
                    env.intersection = each_ic
                    o, s = env.getCurrentOccasion()
                    a = RL.choose_action(s, o) # get action
                    env.prolongTL(a)
                
        if step % VERIFY_INTERVAL == 0:
            env.calWaitingTime()

        step += 1
        env.conn.simulationStep()
    env.conn.close()
    sys.stdout.flush()
    
    print(round(np.mean(env.waitingtime['Sum']), 2), len(env.waitingtime['Sum']))
    return env.waitingtime







