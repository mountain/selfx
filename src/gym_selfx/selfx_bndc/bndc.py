# -*- coding: utf-8 -*-

import gym_selfx.selfx.selfx as selfx


# Award system
NOGAIN = 10
GAIN = 11
PUNISHED = 12

# Agent action
NOOP = 100


class SelfxBoundaryCandyEnvironment(selfx.SelfxEnvironment):
    def __init__(self):
        super(SelfxBoundaryCandyEnvironment, self).__init__()
