
import sys
import numpy as np
import matplotlib.pyplot as plt

from traci_env_single import myTraci
from global_var import VERIFY_INTERVAL, testCmd, actCmd, current_time

def plotTestResult(rl, fixed, actuated, simulation_path):
    plt.ioff()
    plt.figure(figsize = (12,6),dpi = 400)
    plt.plot(np.arange(len(rl))*VERIFY_INTERVAL, np.array(rl), color = '#FF341F', label = 'RL') # red
    plt.plot(np.arange(len(fixed))*VERIFY_INTERVAL, np.array(fixed), color = '#006ED4', label= 'Fixed') # blue
    plt.plot(np.arange(len(actuated))*VERIFY_INTERVAL, np.array(actuated), color = '#30AA52', label= 'Adaptive') #green
    plt.xlabel('Time', fontsize = 12)
    plt.ylabel('Total waiting time', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(loc=4)
    plt.savefig(simulation_path + 'verify_single.png')
    # plt.show()
    plt.close('all')


def testAgent(sim_type, RL):
    if sim_type == 'rl':
        env = myTraci(sumo_cmd = testCmd)
    elif sim_type =='actuated':
        env = myTraci(sumo_cmd = actCmd)
    else:
        env = myTraci(sumo_cmd = testCmd)
        
    env.warmUp(category = 'test')
    # The traffic signal program in testing cases could be different in length from training case
    step = env.conn.simulation.getTime()
    print('Simulating for {}.'.format(sim_type))
    
    while step < env.duration:
        if sim_type == 'rl':             
            env.getUnutilizedGreen(step)
            if env.isCheckpoint():
                o, s = env.getCurrentOccasion()
                a = RL.chooseAction(s, o)
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
    from global_var import green_states, WORKSPACE
    
    trained_path = '{}/results/{}/'.format(WORKSPACE, 'p5i3g0')
    qtable_path = trained_path + 'qtable.csv'
    RL = QLearningTable(list(range(len(green_states))))
    RL.feedQTable(qtable_path)
    RL.epsilon = 1
    fixed,rl,actuated = testAgent('fixed', RL), testAgent('rl', RL), testAgent('actuated', RL)
    plotTestResult(rl, fixed, actuated, trained_path)

