import numpy as np
map = {0:'R',1:'P',2:'S'}
mapInverse = {'R':0,'P':1,'S':2}
rewards = {'R':{'R':-2,'P':-5,'S':10},'P':{'R':10,'P':-2,'S':-5},'S':  {'R':-5,'P':10,'S':-2}}
STATES = 3    #opponent     
ACTIONS = 3  #player
# LEARNING_RATE = 0.75
# GAMMA = 0.7
LEARNING_RATE = 0.95    # a higher rate a faster the model learn
GAMMA = 0.75
Q = np.zeros((STATES,ACTIONS))
epsilon = 0.9
EPISODES= 999
best = {'R':2,'P':0,'S':1}


# map = {0:'R',1:'P',2:'S'}
# mapInverse = {'R':0,'P':1,'S':2}
# rewards = {'R':{'R':-3,'P':-6,'S':10},'P':{'R':10,'P':-3,'S':-6},'S':  {'R':-6,'P':10,'S':-3}}
# STATES = 3    #opponent     
# ACTIONS = 3  #player
# # LEARNING_RATE = 0.75
# # GAMMA = 0.7
# LEARNING_RATE = 0.95    # a higher rate a faster the model learn
# GAMMA = 0.78
# Q = np.zeros((STATES,ACTIONS))
# epsilon = 0.8
# EPISODES= 999
# best = {'R':2,'P':0,'S':1}