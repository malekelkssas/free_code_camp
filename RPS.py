# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import myGlobal

def player(prev_play, opponent_history=[]):
  if check(myGlobal.EPISODES):
    opponent_history = []
  if len(opponent_history)==0:
    opponent_history.append(0)
    return myGlobal.map[0]
  myGlobal.EPISODES -=1
  action=-1
  state = opponent_history[-1]  #my prev play
  reward = myGlobal.rewards[myGlobal.map[state]][prev_play]
  if np.random.uniform(0,1)<myGlobal.epsilon :
    action = int(np.random.uniform(0,3))
  else:
    action = np.argmax(myGlobal.Q[state,:]) 
  next_state = action
  myGlobal.Q[state,action] = myGlobal.Q[state,action] + myGlobal.LEARNING_RATE * (reward + myGlobal.GAMMA *  np.max(myGlobal.Q[next_state, :]) - myGlobal.Q[state,action])
  myGlobal.epsilon-=0.0001
  opponent_history.append(action)
  return myGlobal.map[action]

  
def check(E):
  if E==0 :
    myGlobal.Q = np.zeros((myGlobal.STATES,myGlobal.ACTIONS))
    myGlobal.epsilon = 0.9
    myGlobal.EPISODES= 999
    return True
  return False

