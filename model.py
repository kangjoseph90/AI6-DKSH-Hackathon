# File name : model.py
# Author : Ted Song
# Last Updated : 2021-09-17

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from collections import deque

class DQN:
    def __init__(self):
    	# Setting Hyper Parameters
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 200
        self.gamma = 0.95
        self.lr = 1e-3
        self.batch_size = 64

        ## Layer Model ##
        self.linear_1 = nn.Linear(in_features=7, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=3)

        self.model = nn.Sequential(
        	self.linear_1, nn.ReLU(), self.linear_2
    	)

        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.steps_done = 0
        self.epi_for_memory = deque(maxlen=10000)
    
    def memorize(self, state, action, reward, next_state):
        # self.epi_for_memory = [(상태, 행동, 보상, 다음 상태)...]
        self.epi_for_memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))
    
    def select_acttion(self, state):
        # 지수함수를 이용하여 Epsilon의 값이 점점 Decay됨
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # (Decaying) Epsilon-Greedy Algorithm
        if np.random.random() > epsilon_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[np.random.randint(2, size=1)]])

    def train(self):
        # 저장해 놓은 메모리 사이즈가 배치 사이즈보다 작으면 학습 안함
        if len(self.epi_for_memory) < self.batch_size:
            return

        # 저장해 놓은 메모리 사이즈를 배치 사이즈만큼 Sample로 뽑음
        batch = np.random.choice(self.epi_for_memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)
        
        max_next_q = self.model(next_states).detach().max(1)[0]

        # Bellman equation
        # Q(s, a) := R + discount * max(Q(s', a))
        expected_q = rewards + (self.gamma * max_next_q)
        
        # loss func = MSE (Mean Square Error)
        # [before update current Q-value - after update currnet Q-value]^2
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()