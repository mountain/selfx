# -*- coding: utf-8 -*-

from gym.envs.registration import register


register(
    id='selfx-bounday-candy-v0',
    entry_point='gym_selfx.envs:SelfxBoundaryCandyEnv',
)
