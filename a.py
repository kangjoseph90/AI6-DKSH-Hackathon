import torch
import numpy as np
from model import DQN

agent = DQN()
state = torch.FloatTensor([-0.1699, -0.0758,  0.5065, -0.1699, -0.1699, -0.1699,])
print(state.reshape(2, 3).shape)
# print(np.argmax(state.data).view(1, 1))