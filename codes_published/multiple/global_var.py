import os
import re
import pandas as pd
import time

# In multiple-intersection scenario we only deploy the trained agent into each intersection
# So variable SIM_ITERATION is no longer needed in this case

SIM_DURATION = 7200
VERIFY_INTERVAL = 30
current_time = time.strftime('%m%d%H%M',time.localtime(time.time()))

SCENARIO = 'multiple'
WORKSPACE = os.path.realpath('run.py').split('codes')[0]
DATASPACE = '{}/data/{}'.format(WORKSPACE, SCENARIO)

TLDB = pd.read_csv('{}/tldb_{}.csv'.format(DATASPACE, SCENARIO), sep = ';')
green_states = re.findall(re.compile('\w*G\w*') , str(TLDB['Sumo_code']))

sumoBinary = "{}/bin/sumo".format(os.environ['SUMO_HOME'])
trainCmd = [sumoBinary, "-c", "{}/train_{}.sumocfg".format(DATASPACE, SCENARIO)]
testCmd = [sumoBinary, "-c", "{}/test_{}.sumocfg".format(DATASPACE, SCENARIO)]
actCmd = [sumoBinary, "-c", "{}/actuated_{}.sumocfg".format(DATASPACE, SCENARIO)]




