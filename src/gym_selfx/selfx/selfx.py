# -*- coding: utf-8 -*-

import random
import numpy as np


# Game system
IN_GAME = 0
OUT_GAME = 1


# Action system
NOOP = 10
QUIT = 11


class SelfxToolkit:

    def build_world(self):
        return SelfxWorld()

    def build_rules(self):
        return SelfxGameRules()

    def build_agent(self, inner_world):
        return SelfxAgent(inner_world)

    def build_scope(self):
        return SelfxScope()


class SelfxWorld:
    def __init__(self):
        self.status = IN_GAME
        self.scores = 100

        self.agent = None

        self.step_handlers = []
        self.changed_handlers = []

    def availabe_actions(self):
        return [NOOP, QUIT]

    def act(self, actions):
        return random.sample(actions, 1)

    def state(self):
        return self.status

    def reward(self):
        return self.scores

    def reset(self):
        pass

    def step(self, action):
        if action == NOOP:
            self.fire_step_event()

        if action == QUIT:
            self.status = OUT_GAME

    def render(self, mode='human', close=False):
        return np.zeros([100, 100, 3], dtype='uint8')

    def add_agent(self, agent):
        self.agent = agent

    def add_step_handler(self, handler):
        if handler not in self.step_handlers:
            self.step_handlers.append(handler)

    def add_change_handler(self, handler):
        if handler not in self.changed_handlers:
            self.changed_handlers.append(handler)

    def fire_step_event(self, **pwargs):
        for h in self.step_handlers:
            h.on_world_stepped(self, **pwargs)

    def fire_changed_event(self, **pwargs):
        for h in self.changed_handlers:
            h.on_world_changed(self, **pwargs)


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

        self.fire_changed_event(scope=snapshot * mask)

    def fire_changed_event(self, **pwargs):
        for h in self.handlers:
            h.on_scope_changed(self, **pwargs)


