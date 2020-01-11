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
        super(SimpleDQN, self).__init__()
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

        self.co1 = torch.scalar_tensor(1.0, dtype=torch.float32)
        self.co2 = torch.scalar_tensor(1.0, dtype=torch.float32)
        self.co3 = torch.scalar_tensor(1.0, dtype=torch.float32)

    def crossover(self, another):
        coeff = torch.sigmoid(self.co1 + another.co1)
        self.conv1.weight = self.conv1.weight * coeff + another.conv1.weight * (1 - coeff)
        self.conv1.bias =  self.conv1.bias * coeff + another.conv1.bias * (1 - coeff)
        coeff = torch.sigmoid(self.co2 + another.co2)
        self.conv2.weight = self.conv2.weight * coeff + another.conv2.weight * (1 - coeff)
        self.conv2.bias = self.conv2.bias * coeff + another.conv2.bias * (1 - coeff)
        coeff = torch.sigmoid(self.co3 + another.co3)
        self.conv3.weight = self.conv3.weight * coeff + another.conv3.weight * (1 - coeff)
        self.conv3.bias = self.conv3.bias * coeff + another.conv3.bias * (1 - coeff)

        if random.random() > 0.90:
            self.co1 = torch.tanh(self.co1 - another.co1) * random.random()
            self.co2 = torch.tanh(self.co2 - another.co2) * random.random()
            self.co3 = torch.tanh(self.co3 - another.co3) * random.random()

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
    lmt = screen_height // 3 * 2
    screen = np.ascontiguousarray(screen[0:lmt, :, :], dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)
