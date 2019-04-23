import matplotlib.pyplot as plt
import numpy as np
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CnnActorCriticNetwork, RNDModel
from envs import *
from utils import RunningMeanStd, RewardForwardFilter
from arguments import get_args

from PIL import Image

args = get_args()
device = torch.device('cuda' if args.cuda else 'cpu')

env = gym.make(args.env_name)

input_size = env.observation_space.shape  # 4
output_size = env.action_space.n  # 2
print(env.action_space)

s0 = env.reset()
plt.show()
s, reward, done, info = env.step(1)
s, reward, done, info = env.step(1)
s, reward, done, info = env.step(1)
s, reward, done, info = env.step(1)
plt.show()

print(reward)
print(done)
print(info)

frame = Image.fromarray(s).convert('L')
frame = np.array(frame.resize((84, 84)))
plt.imshow(frame.astype(np.float32))
plt.show()

frame = Image.fromarray(s).convert('L')
frame = np.array(frame.resize((52, 52), Image.BILINEAR))
print(frame.shape)
plt.imshow(frame.astype(np.float32))
plt.show()

if 'Breakout' in args.env_name:
    output_size -= 1

env.close()
