
import sys
import numpy as np
import matplotlib.pyplot as plt

from traci_env_single import myTraci
from global_var import VERIFY_INTERVAL, testCmd, actCmd, current_time

def test_plot(rl, fixed, actuated, simulation_path):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 120)
    
    plt.plot(np.arange(len(rl))*VERIFY_INTERVAL, np.array(rl), color = 'red', label = 'RL')
    plt.plot(np.arange(len(fixed))*VERIFY_INTERVAL, np.array(fixed), label= 'Fixed')
    plt.plot(np.arange(len(actuated))*VERIFY_INTERVAL, np.array(actuated), label= 'Actuated')
    
    plt.xlabel('Time')
    plt.ylabel('Total delay time')
    plt.legend(loc=4)
    plt.savefig(simulation_path + 'verify_single.png')
#    plt.show()


def test_agent(sim_type, RL):
    if sim_type == 'rl':
        env = myTraci(sumo_cmd = testCmd)
    elif sim_type =='actuated':
        env = myTraci(sumo_cmd = actCmd)
    else:
        env = myTraci(sumo_cmd = testCmd)
        
    env.warmUp(category = 'test')# because in test cases we use different TL program and cycle time is different
    step = env.conn.simulation.getTime()
    print('Simulating for {}.'.format(sim_type))
    
    while step < env.episodes:
        if sim_type == 'rl':             
            env.getUtility(step) # right after warmup
            if env.isCheckpoint():
                o, s = env.getCurrentOccasion()
                a = RL.choose_action(s, o) # get action
                env.prolongTL(a)
                
        if step % VERIFY_INTERVAL == 0:
            env.calWaitingTime()
        step += 1
        env.conn.simulationStep()
        
    env.conn.close()
    sys.stdout.flush()
    
    print(round(np.mean(env.waitingtime), 2), len(env.waitingtime))
    return env.waitingtime


if __name__ == '__main__':
    from RL_brain import QLearningTable
    from global_var import green_states
    
    trained_path = '/Users/eriti/Desktop/vgit/RL_traffic_control/results/03151635/'
    qtable_path = trained_path + 'qtable.csv'
    RL = QLearningTable(list(range(len(green_states))))
    RL.feed_qtable(qtable_path)
    RL.epsilon = 1
    fixed,rl,actuated = test_agent('fixed', RL), test_agent('rl', RL), test_agent('actuated', RL)
    test_plot(rl, fixed, actuated, trained_path)

