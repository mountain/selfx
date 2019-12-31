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
from gym_selfx.nn.senet import SEResNet, BasicResidualSEBlock


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


class SimpleDQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


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
