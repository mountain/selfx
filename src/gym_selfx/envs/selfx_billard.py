# -*- coding: utf-8 -*-

import logging


from gym_selfx.selfx_billard import billard
from gym_selfx.envs.selfx_env import SelfXEnv


logger = logging.getLogger(__name__)


class SelfxBillardEnv(SelfXEnv):
    def __init__(self):
        super(SelfxBillardEnv, self).__init__(billard.SelfxBillardToolkit())
