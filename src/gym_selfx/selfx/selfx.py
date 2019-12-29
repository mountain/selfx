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


class SelfxAffordable:
    def __init__(self, ctx, name):
        self._name = name
        self.ctx = ctx
        self.changed_handlers = []

        self._state = self.available_states()[0]
        self._action =  self.available_actions()[0]

    def name(self):
        return self._name

    def subaffordables(self):
        return ()

    def available_actions(self):
        return ()

    def available_states(self):
        return ()

    def action(self):
        return self._action

    def state(self):
        return self._state

    def reset(self):
        for sub in self.subaffordables():
            sub.reset()

    def on_stepped(self, src, **pwargs):
        pass

    def add_change_handler(self, handler):
        if handler not in self.changed_handlers:
            self.changed_handlers.append(handler)

    def fire_changed_event(self, **pwargs):
        for h in self.changed_handlers:
            h.on_changed(self, **pwargs)


class SelfxGame:
    def __init__(self, ctx):
        self.ctx = ctx
        self.affordables = []
        self.actions_list = []
        self.states_list = []

    def add_affordable(self, affordable):
        for sub in affordable.subaffordables():
            self.affordables.append(sub)
        self.affordables.append(affordable)

        self.ctx['outer'].add_step_handler(affordable)
        affordable.add_change_handler(self.ctx['outer'])
        for sub in affordable.subaffordables():
            self.ctx['outer'].add_step_handler(sub)
            sub.add_change_handler(self.ctx['outer'])

        import itertools, collections

        fields = [a.name() for a in self.affordables]

        self.actions_list = [collections.namedtuple('Action', fields)._make(actions)
                             for actions in itertools.product(*[a.available_actions() for a in self.affordables])]

        self.states_list = [collections.namedtuple('State', fields)._make(states)
                            for states in itertools.product(*[a.available_states() for a in self.affordables])]

    def action_space(self):
        return self.actions_list

    def state_space(self):
        return self.states_list

    def action(self):
        import collections

        fields = [a.name() for a in self.affordables]
        a = collections.namedtuple('Action', fields)._make([a.action() for a in self.affordables])
        return a

    def state(self):
        import collections

        fields = [a.name() for a in self.affordables]
        s = collections.namedtuple('State', fields)._make([a.state() for a in self.affordables])
        return s

    def act(self, observation, reward, done):
        return random.sample(self.action_space(), 1)

    def reward(self):
        return 0.0


class SelfxWorld(SelfxAffordable):
    def __init__(self, ctx, name):
        super(SelfxWorld, self).__init__(ctx, name)
        self.step_handlers = []

    def availabe_actions(self):
        return (NOOP, QUIT)

    def availabe_states(self):
        return (IN_GAME, OUT_GAME)

    def reset(self):
        pass

    def step(self, action):
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


class SelfxAgent(SelfxAffordable):
    def __init__(self, ctx):
        super(SelfxAgent, self).__init__(ctx, 'monster')
        self.inner_world = ctx['inner']

    def get_center(self):
        return 0, 0

    def add_move_handler(self, handler):
        pass


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
