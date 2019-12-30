# -*- coding: utf-8 -*-

"""
This code is originally derived from
 Reinforcement Learning (DQN) Tutorial by `Adam Paszke <https://github.com/apaszke>`

"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
from PIL import Image
from senet import SEResNet, BasicResidualSEBlock


Transition = namedtuple('Transition', (
    'state', 'action', 'next_state', 'reward'
))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.net = SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], class_num=outputs)

    def forward(self, x):
        x1 = x[:, 0:1, 0:64]
        x2 = x[:, 0:1, 64:128]
        x3 = x[:, 1:2, 0:64]
        x4 = x[:, 1:2, 64:128]
        x5 = x[:, 2:3, 0:64]
        x6 = x[:, 2:3, 64:128]
        return F.log_softmax(self.net(torch.cat((x1, x2, x3, x4, x5, x6), dim=1)), dim=1)


resize = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64), interpolation=Image.CUBIC),
    T.ToTensor()
])


def get_screen(env, device):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)
