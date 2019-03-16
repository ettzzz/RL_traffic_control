import os
import sys
import numpy as np

from global_var import WORKSPACE, VERIFY_INTERVAL, SIM_ITERATIONS, current_time
from traci_env_single import myTraci


def generate_path(time_tag):
    rst_path = '{}/results/'.format(WORKSPACE)
    sim_path = rst_path + time_tag + '/'
    
    try:
        os.makedirs('{}/trashcan/'.format(WORKSPACE))
        os.makedirs(rst_path)
        print('Creating /results/ folder successfully.')
    except:
        pass
    
    if os.path.exists(sim_path) == False:
        os.makedirs(sim_path)
    else:
        print('There is already a folder named {}.'.format(time_tag))
    
    return rst_path, sim_path


def write_log(RL_params, rst_path, sim_path, clean = False):
    
    with open ('{}/params.txt'.format(sim_path),'w') as log:
        log.write('Simulation configurations' + '\n')
        log.write('Script path: {}\n'.format(os.path.realpath('run.py')))
        log.write('RL iterations: {}\n'.format(str(SIM_ITERATIONS)))
        log.write('RL params: \
                  \n\tLearning rate: {}\
                  \n\tReward decay: {}\
                  \n\tepsilon max: {}\
                  \n\tepsilon increment: {}'.format(RL_params['lr'], RL_params['gamma'], RL_params['e_max'], RL_params['e_inc']))
        
    # integrated with function of cleaning empty folders
    if clean == True:
        trashcan_path = WORKSPACE + '/trashcan/'
        results_list = os.listdir(rst_path)
        for each_file in results_list:
            absolute_path = rst_path + each_file
            if os.path.isdir(absolute_path) and len(os.listdir(absolute_path)) == 0 and int(each_file[:8]) < int(current_time[:8]):
                print('folder {} will be removed to /trashcan/ folder.'.format(each_file))
                os.system('mv {} {}'.format(absolute_path, trashcan_path))
                # shutil.move(absolute_path, trashcan_path) # for general use
                
                    

def train_agent(RL, rst_path, sim_path):
    scatter = []
    for episode in range(SIM_ITERATIONS): 
        
        env = myTraci()
        env.warmUp()
        step = env.conn.simulation.getTime()
        print('Training episode {}/{}.'.format(str(episode+1), SIM_ITERATIONS))
        while step < env.episodes:
#        while step < 1200:
            env.getUtility(step)
            if env.isCheckpoint():
                r = env.getReward()
                o, s = env.getCurrentOccasion()
                a = RL.choose_action(s, o) 
                env.prolongTL(a)
                RL.write_memory_record(o, s, a, r)
                RL.write_log(episode, step, s, o, a, r)
            else:
                pass
            if step % VERIFY_INTERVAL == 0:
                env.calWaitingTime()
            step += 1
            env.conn.simulationStep()

        avg_wt = round(np.mean(list(env.waitingtime)),2)
        cr = round(sum(RL.learnlog[(RL.learnlog['epi'] == episode)]['r']))
        print('average waiting time:', avg_wt, 'cumulative reward:', cr)
        scatter.append(avg_wt)
        
        env.conn.close()
        sys.stdout.flush()
        
    env.plotScatter(scatter, sim_path)
    RL.learnlog.to_csv('{}/learnlog.csv'.format(sim_path), index = False, sep = ';')
    

  

