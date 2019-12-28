# -*- coding: utf-8 -*-

import random
import numpy as np


IN_GAME = 'in_game'
OUT_GAME = 'out_game'

NOOP = 'noop'
QUIT = 'quit'


class SelfxToolkit:
    def build_game(self, ctx):
        return SelfxGame(ctx)

    def build_inner_world(self, ctx, name):
        return SelfxWorld(ctx, 'inner')

    def build_outer_world(self, ctx):
        return SelfxWorld(ctx, 'outer')

    def build_agent(self, ctx):
        return SelfxAgent(ctx)

    def build_scope(self, ctx):
        return SelfxScope(ctx)


class SelfxGame:
    def __init__(self, ctx):
        self.ctx = ctx
        self.affordables = []
        self.actions_list = []
        self.states_list = []

    def add_affordable(self, affordable):
        self.affordables.append(affordable)

        import itertools, collections

        fields = [a.name() for a in self.affordables]

        self.actions_list = [collections.namedtuple('Action', fields, defaults=action)
                for action in itertools.product(*[a.available_actions() for a in self.affordables])]

        self.states_list = [collections.namedtuple('State', fields, defaults=states)
                for states in itertools.product(*[a.available_states() for a in self.affordables])]

    def available_actions(self):
        return self.actions_list

    def available_states(self):
        return self.states_list

    def action(self):
        import collections

        fields = [a.name() for a in self.affordables]
        a = collections.namedtuple('action', fields, defaults=[a.action() for a in self.affordables])
        return self.actions_list.index(a)

    def state(self):
        import collections

        fields = [a.name() for a in self.affordables]
        s = collections.namedtuple('state', fields, defaults=[a.state() for a in self.affordables])
        return self.states_list.index(s)

    def enforce_on(self, world):
        pass


class SelfxWorld:
    def __init__(self, ctx, aname):
        self.ctx = ctx

        self._name = aname
        self.status = IN_GAME
        self.scores = 100

        self.agent = None

        self.step_handlers = []
        self.changed_handlers = []

    def name(self):
        return self._name

    def availabe_actions(self):
        return (NOOP, QUIT)

    def availabe_states(self):
        return (IN_GAME, OUT_GAME)

    def act(self, actions):
        return random.sample(actions, 1)

    def state(self):
        return self.status

    def reward(self):
        return self.scores

    def reset(self):
        pass

    def step(self, action):
        if action[self._name] == NOOP:
            self.fire_step_event()

        if action[self._name] == QUIT:
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


class SelfxAgent:
    def __init__(self, ctx):
        self.ctx = ctx
        self.inner_world = ctx['inner']

    def get_center(self):
        return 0, 0

    def act(self, observation, reward, done):
        return [random.sample(self.available_actions(), 1), random.sample(self.inner_world.availabe_actions(), 1)]

    def add_move_handler(self, handler):
        pass


class SelfxScope:
    def __init__(self, ctx):
        self.ctx = ctx

    def get_mask(self, snapshot):
        return np.ones(snapshot.shape)

    def add_changed_handler(self, h):
        pass

    def on_agent_move(self, agent):
        self.centerx, self.centery = agent.get_center()

    def on_world_stepped(self, world):
        snapshot = world.get_snapshot()
        mask = self.get_mask(snapshot)

        self.fire_changed_event(scope=snapshot * mask)

    def fire_changed_event(self, **pwargs):
        for h in self.handlers:
            h.on_scope_changed(self, **pwargs)


