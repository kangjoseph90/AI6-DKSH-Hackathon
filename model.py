# File name : model.py
# Author : Ted Song
# Last Updated : 2021-09-17

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np
from collections import deque

class DQN:
    def __init__(self, episode):
    	# Setting Hyper Parameters
        self.epsilon_start = 0.95
        self.epsilon_end = 0.05
        self.epsilon_decay = episode
        self.gamma = 0.95
        self.lr = 1e-3
        self.batch_size = 256

        self.model = nn.Sequential(
            nn.Linear(in_features=6, out_features=256),
            # nn.LSTM(input_size=256, hidden_size=3, num_layers=1),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=3)
    	)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.epi_for_memory = deque(maxlen=100000)
        self.epsilon_threshold = 0.95


    def printEps(self):
        return self.epsilon_threshold

    def decay_epsilon(self):
        # 지수함수를 이용하여 Epsilon의 값이 점점 Decay됨
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

    def memorize(self, state, action, reward, next_state):
        # self.epi_for_memory = [(상태, 행동, 보상, 다음 상태)...]
        self.epi_for_memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor(next_state)))
    
    def select_action(self, state):
        # (Decaying) Epsilon-Greedy Algorithm
        if torch.rand(1)[0] > self.epsilon_threshold:
            with torch.no_grad():
                return torch.argmax(self.model(state).data).view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def optimize_model(self):
        # 저장해 놓은 메모리 사이즈가 배치 사이즈보다 작으면 학습 안함
        if len(self.epi_for_memory) < self.batch_size:
            return

        # 저장해 놓은 메모리 사이즈를 배치 사이즈만큼 Sample로 뽑음
        batch = random.sample(self.epi_for_memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states).reshape(256, 6)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states).reshape(256, 6)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]

        # Bellman equation
        # Q(s, a) := R + discount * max(Q(s', a))
        expected_q = rewards + (self.gamma * max_next_q)
        
        # loss func = MSE (Mean Square Error)
        # [before update current Q-value - after update currnet Q-value]^2
        self.optimizer.zero_grad()
        loss = self.criterion(current_q.squeeze(), expected_q)
        loss.backward()

        self.optimizer.step()