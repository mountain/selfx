# -*- coding: utf-8 -*-

import gym
import gym_selfx.selfx.selfx as selfx

from gym import utils


import logging
logger = logging.getLogger(__name__)


class SelfXEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = self.init_environment()

        self.inner = self.env.build_world()
        self.outer = self.env.build_world()
        self.rules = self.env.build_rules()
        self.scope = self.env.build_scope()
        self.agent = self.env.build_agent()

        self.outer.add_step_handler(self.scope)

        self.agent.add_handler(self.scope)
        self.scope.add_handler(self.agent)
        self.outer.add_agent(self.agent)

        self.rules.apply_to(self.outer)
        self.outer.add_handler(self.rules)

        self.action_space = [(a, b) for a in self.inner.availabe_actions() for b in self.outer.availabe_actions()]

        self.status = (selfx.IN_GAME, selfx.IN_GAME)

    def __del__(self):
        self.inner.act(selfx.QUIT)
        self.inner.step()
        self.outer.act(selfx.QUIT)
        self.outer.step()

    def init_environment(self):
        return None

    def step(self, action):
        action1, action2 = action

        self.inner.act(action1)
        self.outer.act(action2)

        status1 = self.inner.step()
        status2 = self.outer.step()

        reward = self.outer.get_reward()

        obs1 = self.inner.getState()
        obs2 = self.outer.getState()

        episode_over = (status1 != selfx.IN_GAME) and (status2 != selfx.IN_GAME)

        return (obs1, obs2), reward, episode_over, {}

    def reset(self):
        self.inner.reset()
        self.outer.reset()
        return self.inner.getState(), self.outer.getState()

    def render(self, mode='human', close=False):
        self.inner.render(mode, close)
        self.outer.render(mode, close)
