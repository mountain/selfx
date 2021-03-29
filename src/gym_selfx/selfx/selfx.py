# -*- coding: utf-8 -*-

import random
import numpy as np

from affordable.game import AbstractGame, Affordable


NOOP = 'noop'
IDLE = 'idle'


class SelfxToolkit:
    def build_game(self, ctx):
        return SelfxGame(ctx)

    def build_inner_world(self, ctx, name):
        return SelfxWorld(ctx, 'inner')

    def build_outer_world(self, ctx):
        return SelfxWorld(ctx, 'outer')

    def build_agent(self, ctx, eye):
        return SelfxAgent(ctx, eye)

    def build_eye(self, ctx):
        return SelfxEye(ctx)

    def build_scope(self, ctx):
        return SelfxScope(ctx)


class SelfxGame(AbstractGame):
    def __init__(self, ctx):
        super(SelfxGame, self).__init__(ctx, 'game')
        self.policy = None


class SelfxWorld(Affordable):
    def __init__(self, ctx, name):
        super(SelfxWorld, self).__init__(ctx, name)
        self.step_handlers = []

    def reset(self):
        pass

    def act(self, action):
        self.fire_step_event(action=action)

    def render(self, mode='rgb_array', close=False):
        return np.zeros([100, 100, 3], dtype='uint8')

    def add_agent(self, agent):
        self.agent = agent

    def get_snapshot(self):
        return self.render()

    def add_step_handler(self, handler):
        if handler is not self and handler not in self.step_handlers:
            self.step_handlers.append(handler)

    def fire_step_event(self, **pwargs):
        for h in self.step_handlers:
            h.on_stepped(self, **pwargs)

    def act(self, action):
        self.fire_step_event()


class SelfxAgent(Affordable):
    def __init__(self, ctx, eye):
        super(SelfxAgent, self).__init__(ctx, 'monster')
        self.inner_world = ctx['inner']
        self.eye = eye

    def center(self):
        return 0, 0

    def direction(self):
        return 0, 0

    def add_move_handler(self, handler):
        pass

    def act(self, action):
        pass


class SelfxEye:
    def __init__(self, ctx):
        self.ctx = ctx
        self._state = self.available_states().__next__()

    def name(self):
        return 'eye'

    def view(self, world, center, direction):
        w = world.render()
        return np.zeros(w.shape)

    def available_states(self):
        return '0', '1'

    def state(self):
        return self._state


class SelfxScope:
    def __init__(self, ctx):
        self.ctx = ctx
        self.handlers = []

    def get_mask(self, snapshot):
        return np.ones(snapshot.shape)

    def add_changed_handler(self, h):
        pass

    def on_agent_move(self, agent):
        self.centerx, self.centery = agent.get_center()

    def on_stepped(self, world, **pwargs):
        snapshot = world.get_snapshot()
        mask = self.get_mask(snapshot)

        self.fire_changed_event(scope=snapshot * mask)

    def fire_changed_event(self, **pwargs):
        for h in self.handlers:
            h.on_scope_changed(self, **pwargs)
