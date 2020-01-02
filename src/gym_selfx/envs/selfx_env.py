# -*- coding: utf-8 -*-

import numpy as np
import cv2

import gym
import gym_selfx.selfx.selfx as selfx

from gym import utils


import logging
logger = logging.getLogger(__name__)


class SelfXEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['rgb_array']}

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

        eye = self.toolkit.build_eye(ctx)
        self.agent = self.toolkit.build_agent(ctx, eye)
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
        self.inner.step(action=action)
        self.outer.step(action=action)

        reward = self.game.reward()

        _state = self.state()

        episode_over = self.game.exit_condition() or self.game.force_condition()

        return _state, reward, episode_over, {}

    def reset(self):
        self.inner.reset()
        self.outer.reset()
        self.agent.reset()
        return self.state()

    def render(self, mode='rgb_array', close=False):
        self.inner.drawer.install()
        arr1 = self.inner.render(mode, close)
        self.outer.drawer.install()
        arr2 = self.outer.render(mode, close)
        view = self.agent.eye.view(self.outer, self.agent.center(), self.agent.direction())
        r = cv2.resize(view[:, :, 0], (arr2.shape[0], arr2.shape[1]), interpolation=cv2.INTER_CUBIC).reshape(arr2.shape[0], arr2.shape[1], 1)
        g = cv2.resize(view[:, :, 1], (arr2.shape[0], arr2.shape[1]), interpolation=cv2.INTER_CUBIC).reshape(arr2.shape[0], arr2.shape[1], 1)
        b = cv2.resize(view[:, :, 2], (arr2.shape[0], arr2.shape[1]), interpolation=cv2.INTER_CUBIC).reshape(arr2.shape[0], arr2.shape[1], 1)
        view = np.concatenate([r, g, b], axis=2)

        return np.concatenate([view, arr1, arr2], axis=0)
