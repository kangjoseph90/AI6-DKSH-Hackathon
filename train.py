from model import DQN
from snake import Snake
import torch

score_history = []

for e in range(50):
    state = Snake()
    steps = 0
    while True:
        state = torch.FloatTensor([state])

        action = DQN().select_acttion(state) 
        next_state, reward, done, _ = state.getState()

        if done:
            reward = -1

        DQN().memorize(state, action, reward, next_state)
        DQN().train()

        state = next_state
        steps += 1

        if done:
            print("에피소드:{0} 점수: {1}".format(e, steps))
            score_history.append(steps)
            break