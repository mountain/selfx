# -*- coding: utf-8 -*-

import logging
import numpy as np

from gym_selfx.selfx_billard import billard
from gym_selfx.envs.selfx_env import SelfXEnv


logger = logging.getLogger(__name__)


class SelfxBillardEnv(SelfXEnv):
    def __init__(self):
        super(SelfxBillardEnv, self).__init__(billard.SelfxBillardToolkit())

    def state(self):
        img = self.render(mode='rgb_array', close=False)
        h, w, c = img.shape
        h = h // 3

        r = img[:, :, 0:1].reshape(1, 3 * h, w)
        g = img[:, :, 1:2].reshape(1, 3 * h, w)
        b = img[:, :, 2:3].reshape(1, 3 * h, w)
        r1, r2, r3 = r[:, :h, :], r[:, h:2 * h, :], r[:, 2 * h:, :]
        g1, g2, g3 = g[:, :h, :], g[:, h:2 * h, :], g[:, 2 * h:, :]
        b1, b2, b3 = b[:, :h, :], b[:, h:2 * h, :], b[:, 2 * h:, :]
        return np.concatenate((r1, g1, b1, r2, g2, b2, r3, g3, b3), axis=0)
