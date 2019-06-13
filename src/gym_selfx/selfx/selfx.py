# -*- coding: utf-8 -*-

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


class EventSource:

    def add_listener(self, listener):
        pass

    def fire_event(self, evt):
        pass


class SelfxWorld:
    pass


class SelfxGameRules:
    pass


class SelfxAgent:
    pass


class SelfxScope:
    pass

