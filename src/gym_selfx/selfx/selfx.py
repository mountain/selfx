# -*- coding: utf-8 -*-

import numpy as np


# Game system
QUIT = -1
IN_GAME = 0


class SelfxEnvironment:

    def build_world(self):
        return SelfxWorld()

    def build_rules(self):
        return SelfxGameRules()

    def build_agent(self):
        return SelfxAgent()

    def build_scope(self):
        return SelfxScope()


class SelfxWorld:
    pass


class SelfxGameRules:
    pass


class SelfxAgent:
    pass


class SelfxScope:

    def get_mask(self):
        return np.zeros([100, 100])

    def on_world_step(self, world):
        snapshot = world.get_snapshot()
        mask = self.get_mask()
        idxs = self.get_indexes()
        scope = np.take(snapshot * mask, idxs)

        self.fire_scope_event(self, scope=scope)

    def fire_scope_event(self, src, **pwargs):
        for l in self.listerners:
            l.on_scope_event(src, **pwargs)


