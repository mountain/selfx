# -*- coding: utf-8 -*-

import logging
import math

from gym_selfx.selfx_bndc import bndc
from gym_selfx.envs.selfx_env import SelfXEnv


logger = logging.getLogger(__name__)


class SelfxBoundaryCandyEnv(SelfXEnv):
    def __init__(self):
        super(SelfxBoundaryCandyEnv, self).__init__()

    def init_toolit(self):
        return bndc.SelfxBoundaryCandyToolkit()
