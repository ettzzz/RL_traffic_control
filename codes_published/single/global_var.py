import os
import re
import pandas as pd
import time

SIM_ITERATIONS = 400
SIM_DURATION = 7200
VERIFY_INTERVAL = 30
current_time = time.strftime('%m%d%H%M',time.localtime(time.time()))

SCENARIO = 'single'
WORKSPACE = os.path.realpath('run.py').split('codes')[0]
DATASPACE = '{}/data/{}'.format(WORKSPACE, SCENARIO)

TLDB = pd.read_csv('{}/tldb_{}.csv'.format(DATASPACE, SCENARIO), sep = ';') # Traffic light database,
green_states = re.findall(re.compile('\w*G\w*') , str(TLDB['Sumo_code'])) # All green phases in form of SUMO code

sumoBinary = "{}/bin/sumo".format(os.environ['SUMO_HOME'])
trainCmd = [sumoBinary, "-c", "{}/train_{}.sumocfg".format(DATASPACE, SCENARIO)]
testCmd = [sumoBinary, "-c", "{}/test_{}.sumocfg".format(DATASPACE, SCENARIO)]
actCmd = [sumoBinary, "-c", "{}/actuated_{}.sumocfg".format(DATASPACE, SCENARIO)]
