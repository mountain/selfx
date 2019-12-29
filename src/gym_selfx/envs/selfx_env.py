# -*- coding: utf-8 -*-

import numpy as np
import random

import gym
import gym_selfx.selfx.selfx as selfx

from gym import utils


import logging
logger = logging.getLogger(__name__)


class SelfXEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, toolkit):
        self.toolkit = toolkit

        ctx = {}
        self.game = self.toolkit.build_game(ctx)
        self.inner = self.toolkit.build_inner_world(ctx)
        self.outer = self.toolkit.build_outer_world(ctx)
        self.scope = self.toolkit.build_scope(ctx)
        ctx.update({
            'env': self,
            'game': self.game,
            'inner': self.inner,
            'outer': self.outer,
        })

        self.agent = self.toolkit.build_agent(ctx)
        ctx.update({
            'agent': self.agent,
        })

        self.x_threshold = self.outer.x_threshold
        self.y_threshold = self.outer.y_threshold

        self.game.add_affordable(self.outer)
        self.game.add_affordable(self.agent)

        self.outer.add_step_handler(self.scope)
        self.agent.add_move_handler(self.scope)
        self.scope.add_changed_handler(self.agent)
        self.outer.add_agent(self.agent)

        self.outer.add_change_handler(self.game)

        self.action_space = self.game.action_space()
        self.state_space = self.game.state_space()

    def state(self):
        return self.game.state()

    def step(self, action):
        self.inner.step(action)
        self.outer.step(action)

        reward = self.outer.reward()

        _state = self.state()

        episode_over = (_state.outer != selfx.OUT_GAME) and (_state.inner != selfx.OUT_GAME) and random.random() < 0.005

        return (_state.outer, _state.inner), reward, episode_over, {}

    def reset(self):
        self.inner.reset()
        self.outer.reset()
        return self.inner.state(), self.outer.state()

    def render(self, mode='human', close=False):
        arr1 = self.inner.render(mode, close)
        arr2 = self.outer.render(mode, close)
        return np.concatenate([arr1, arr2], axis=0)
