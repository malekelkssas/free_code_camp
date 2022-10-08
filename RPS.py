# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import myGlobal

def player(prev_play, opponent_history=[]):
  if check(myGlobal.EPISODES):
    opponent_history = []
  action=-1
  myGlobal.EPISODES -=1
  state =-1
  if len(opponent_history) == 0:
    state = int(np.random.uniform(0,3))
    opponent_history.append(state)
    return myGlobal.map[state]
  else:
    state = opponent_history[-1]  #my prev play
    opponent_history.append(state)
    reward = myGlobal.rewards[prev_play][myGlobal.map[state]]
    if np.random.uniform(0,1)<myGlobal.epsilon  :
      action = int(np.random.uniform(0,3))
    else:
      action = np.argmax(myGlobal.Q[myGlobal.mapInverse[prev_play],:]) 
    myGlobal.Q[myGlobal.mapInverse[prev_play],state] = myGlobal.Q[myGlobal.mapInverse[prev_play],state] + myGlobal.LEARNING_RATE * (reward + myGlobal.GAMMA *   np.max(myGlobal.Q[action, :]) - myGlobal.Q[myGlobal.mapInverse[prev_play],state])
    myGlobal.epsilon-=0.0009
    return myGlobal.map[action]
def check(E):
  if E==0:
    myGlobal.Q = np.zeros((myGlobal.STATES,myGlobal.ACTIONS))
    myGlobal.epsilon = 0.9
    myGlobal.EPISODES= 1000
    return True
  return False

