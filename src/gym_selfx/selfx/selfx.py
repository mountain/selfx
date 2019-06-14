# -*- coding: utf-8 -*-

import random
import numpy as np


# Game system
QUIT = -1
IN_GAME = 0


# Action system
NOOP = 10


class SelfxEnvironment:

    def build_world(self):
        return SelfxWorld()

    def build_rules(self):
        return SelfxGameRules()

    def build_agent(self, inner_world):
        return SelfxAgent(inner_world)

    def build_scope(self):
        return SelfxScope()


class SelfxWorld:

    def availabe_actions(self):
        return [NOOP]

    def act(self, actions):
        return NOOP

    def state(self):
        return None

    def reward(self):
        return 0

    def reset(self):
        pass

    def step(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def add_agent(self, agent):
        pass

    def add_step_handler(self, handler):
        pass

    def add_change_handler(self, handler):
        pass


class SelfxGameRules:

    def enforce_on(self, world):
        pass


class SelfxAgent:
    def __init__(self, inner_world):
        self.action_space = [NOOP]
        self.inner_world = inner_world

    def get_center(self):
        return 0, 0

    def act(self, observation, reward, done):
        return [random.sample(self.action_space, 1), random.sample(self.inner_world.availabe_actions(), 1)]

    def add_move_handler(self, handler):
        pass


class SelfxScope:

    def get_mask(self, snapshot):
        return np.ones(snapshot.shape)

    def add_changed_handler(self, h):
        pass

    def on_agent_move(self, agent):
        self.centerx, self.centery = agent.get_center()

    def on_world_step(self, world):
        snapshot = world.get_snapshot()
        mask = self.get_mask(snapshot)

        self.fire_scope_event(self, scope=snapshot * mask)

    def fire_scope_changed_event(self, src, **pwargs):
        for l in self.listerners:
            l.on_scope_changed_event(src, **pwargs)


