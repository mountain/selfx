# -*- coding: utf-8 -*-

import numpy as np
import random

import gym
import gym_selfx.selfx.selfx as selfx

from gym import utils


import logging
logger = logging.getLogger(__name__)


class ActionSpace(list):
    def __init__(self, elments):
        super(ActionSpace, self).__init__(elments)
        self.n = len(elments)


class SelfXEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, toolkit):
        self.toolkit = toolkit

        ctx = {}
        self.inner = self.toolkit.build_world(ctx)
        self.outer = self.toolkit.build_world(ctx)
        self.rules = self.toolkit.build_rules(ctx)
        self.scope = self.toolkit.build_scope(ctx)
        ctx.update({
            'env': self,
            'inner': self.inner,
            'outer': self.outer,
            'rules': self.rules,
        })
        self.agent = self.toolkit.build_agent(ctx)
        ctx.update({
            'agent': self.agent,
        })

        self.x_threshold = self.outer.x_threshold
        self.y_threshold = self.outer.y_threshold

        self.outer.add_step_handler(self.scope)

        self.agent.add_move_handler(self.scope)
        self.scope.add_changed_handler(self.agent)
        self.outer.add_agent(self.agent)

        self.outer.add_change_handler(self.rules)
        self.rules.enforce_on(self.outer)

        self.action_space = ActionSpace([(a, b) for a in self.inner.availabe_actions() for b in self.outer.availabe_actions()])

        self.state = (selfx.IN_GAME, selfx.IN_GAME)

    def __del__(self):
        self.inner.step(selfx.QUIT)
        self.outer.step(selfx.QUIT)

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
