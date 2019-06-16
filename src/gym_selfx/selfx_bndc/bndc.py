# -*- coding: utf-8 -*-

import gym_selfx.selfx.selfx as selfx


# Award system
NOGAIN = 10
GAIN = 11
PUNISHED = 12

# Agent action
NOOP = 100


class SelfxBoundaryCandyToolkit(selfx.SelfxToolkit):
    def __init__(self):
        super(SelfxBoundaryCandyToolkit, self).__init__()
