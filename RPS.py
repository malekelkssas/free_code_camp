# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
# try hidden markov model nexttime

import numpy as np
import myGlobal


def player(prev_play, opponent_history=[]):
  # print('myGlobal.EPISODES: ',myGlobal.EPISODES)
  if check(myGlobal.EPISODES):
    opponent_history = []
  if len(opponent_history)==0:
    state = 0
    opponent_history.append(state)
    return myGlobal.map[state]
  myGlobal.EPISODES -=1
  action=-1
  state = opponent_history[-1]  #my prev play
  reward = myGlobal.rewards[myGlobal.map[state]][prev_play]
  action = np.argmax(myGlobal.Q[state,:]) 
  next_state = action
  myGlobal.Q[state,action] = myGlobal.Q[state,action] +   myGlobal.LEARNING_RATE * (reward + myGlobal.GAMMA *  np.max(myGlobal.Q[next_state, :]) - np.absolute(myGlobal.Q[state,action]))
  opponent_history.append(action)
  myGlobal.epsilon-=0.0001
  return myGlobal.map[action]

  
def check(E):
  if E==0:
    myGlobal.Q = np.zeros((myGlobal.STATES,myGlobal.ACTIONS))
    myGlobal.epsilon = 0.75
    myGlobal.EPISODES= 999
    return True
  return False

