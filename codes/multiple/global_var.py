import os
import re
import pandas as pd
import time

SCENARIO = 'multiple'
WORKSPACE = os.path.realpath('run.py').split('codes')[0]
DATASPACE = '{}/data/{}'.format(WORKSPACE, SCENARIO)

SIM_ITERATIONS = 2
SIM_LENGTH = 7200
VERIFY_INTERVAL = 10
GEN_ROUTE = False

TLDB = pd.read_csv('{}/tldb_{}.csv'.format(DATASPACE, SCENARIO), sep = ';')
green_states = re.findall(re.compile('\w*G\w*') , str(TLDB['Sumo_code']))

sumoBinary = "{}/bin/sumo".format(os.environ['SUMO_HOME'])
trainCmd = [sumoBinary, "-c", "{}/train_{}.sumocfg".format(DATASPACE, SCENARIO)]
testCmd = [sumoBinary, "-c", "{}/test_{}.sumocfg".format(DATASPACE, SCENARIO)]
actCmd = [sumoBinary, "-c", "{}/actuated_{}.sumocfg".format(DATASPACE, SCENARIO)]

current_time = time.strftime('%m%d%H%M',time.localtime(time.time()))


