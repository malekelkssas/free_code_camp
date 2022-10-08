import numpy as np
map = {0:'R',1:'P',2:'S'}
mapInverse = {'R':0,'P':1,'S':2}
rewards = {'R':{'R':-0.5,'P':-0.9,'S':0.9},'P':{'R':0.9,'P':-0.5,'S':-0.9},'S':  {'R':-0.9,'P':0.9,'S':-0.5}}
STATES = 3    #opponent     
ACTIONS = 3  #player
LEARNING_RATE = 0.75
GAMMA = 0.7
Q = np.zeros((STATES,ACTIONS))
epsilon = 0.9
EPISODES= 1000