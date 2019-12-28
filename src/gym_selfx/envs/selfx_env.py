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

        self.game.add_affordable(self.agent)
        self.game.add_affordable(self.outer)
        self.game.add_affordable(self.inner)

        self.x_threshold = self.outer.x_threshold
        self.y_threshold = self.outer.y_threshold

        self.outer.add_step_handler(self.scope)

        self.agent.add_move_handler(self.scope)
        self.scope.add_changed_handler(self.agent)
        self.outer.add_agent(self.agent)

        self.outer.add_change_handler(self.game)
        self.game.enforce_on(self.outer)

        self.action_space = self.game.available_actions()
        self.state_space = self.game.available_states()

    def __del__(self):
        self.inner.step(-1)
        self.outer.step(-1)

    def state(self):
        return self.game.state()

    def step(self, action):
        action1, action2 = action

        self.inner.act(action1)
        self.outer.act(action2)

        status1 = self.inner.step(action1)
        status2 = self.outer.step(action2)

        reward = self.outer.reward()

        obs1 = self.inner.state()
        obs2 = self.outer.state()

        episode_over = (status1 != selfx.OUT_GAME) and (status2 != selfx.OUT_GAME) and random.random() < 0.01

        return (obs1, obs2), reward, episode_over, {}

    def reset(self):
        self.inner.reset()
        self.outer.reset()
        return self.inner.state(), self.outer.state()

    def render(self, mode='human', close=False):
        arr1 = self.inner.render(mode, close)
        arr2 = self.outer.render(mode, close)
        return np.concatenate([arr1, arr2], axis=0)
