# -*- coding: utf-8 -*-

import random
import numpy as np

import gym_selfx.selfx.selfx as selfx

from gym_selfx.render.draw import OpencvDrawFuncs
from Box2D.Box2D import (b2PolygonShape as polygonShape, b2CircleShape as circleShape, b2World)


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS


# Award system
NOGAIN = 10
GAIN = 11
PUNISHED = 12

# Agent action


class SelfxBillardToolkit(selfx.SelfxToolkit):
    def __init__(self):
        super(SelfxBillardToolkit, self).__init__()

    def build_world(self, ctx):
        return SelfxBillardWorld(ctx)

    def build_rules(self, ctx):
        return SelfxBillardGameRules(ctx)

    def build_agent(self, inner_world):
        return SelfxBillardAgent(inner_world)

    def build_scope(self, ctx):
        return SelfxBillardScope(ctx)


class SelfxBillardWorld(selfx.SelfxWorld):
    def __init__(self, ctx):
        super(SelfxBillardWorld, self).__init__(ctx)
        self.drawer = OpencvDrawFuncs(w=1029, h=645, ppm=0.9)
        self.drawer.install()

        self.b2 = b2World(gravity=(0, 0), doSleep=True)
        self.left = self.b2.CreateStaticBody(
            position=(2, 323),
            shapes=polygonShape(box=(1, 641)),
            linearDamping=0.0,
            bullet=True,
        ).CreatePolygonFixture(box=(1, 641))
        self.right = self.b2.CreateStaticBody(
            position=(1027, 323),
            shapes=polygonShape(box=(1, 641)),
            linearDamping=0.0,
            bullet=True,
        ).CreatePolygonFixture(box=(1, 641))
        self.top = self.b2.CreateStaticBody(
            position=(515, 643),
            shapes=polygonShape(box=(1025, 1)),
            linearDamping=0.0,
            bullet=True,
        ).CreatePolygonFixture(box=(1025, 1))
        self.bottom = self.b2.CreateStaticBody(
            position=(515, 2),
            shapes=polygonShape(box=(1025, 1)),
            linearDamping=0.0,
            bullet=True,
        ).CreatePolygonFixture(box=(1025, 1))

        for _ in range(20):
            self.b2.CreateStaticBody(
                position=(random.random() * 1025, random.random() * 641),
                shapes=circleShape(radius=random.random() * 50),
                linearDamping=0.0,
                bullet=True,
            )

        angle = random.random() * 360
        alpha = np.deg2rad(angle)
        self.body = self.b2.CreateDynamicBody(
            position=(513, 321),
            angle=alpha,
            linearVelocity=((random.random() - 0.5) * 500, (random.random() - 0.5) * 500),
            linearDamping=0.0,
            bullet=True,
        )
        self.body.CreateCircleFixture(radius=5.0, density=1, friction=0.0,)
        self.tail = self.b2.CreateDynamicBody(
            position=(513 + 5.0 * np.cos(alpha), 321 + 5.0 * np.sin(alpha)),
            angle=alpha,
            linearVelocity=((random.random() - 0.5) * 500, (random.random() - 0.5) * 500),
            linearDamping=0.0,
        )
        self.tail.CreatePolygonFixture(box=(5, 0.1), density=0.3, friction=0.0,)
        self.monster = self.b2.CreateWeldJoint(
            bodyA=self.body,
            bodyB=self.tail,
            anchor=self.body.worldCenter,
        )

    def act(self, actions):
        if self.scores <= 0:
            return selfx.QUIT
        else:
            return selfx.NOOP

    def step(self, action):
        if random.random() > 0.90:
            self.b2.CreateDynamicBody(
                position=(random.random() * 1025, random.random() * 641),
                linearVelocity=((random.random() - 0.5) * 500, (random.random() - 0.5) * 500),
                angle=random.random() * 360,
                linearDamping=0.0,
                bullet=True,
            ).CreateCircleFixture(radius=5.0, density=1, friction=0.3)

        self.b2.Step(TIME_STEP, 10, 10)

        if action == selfx.NOOP:
            self.scores -= 1.0
            self.fire_step_event()

        if action == selfx.QUIT:
            self.status = selfx.OUT_GAME

    def reset(self):
        self.drawer.clear_screen()

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            self.drawer.clear_screen()
            self.drawer.draw_world(self.b2)
            return 2 * self.drawer.screen
        else:
            return None


class SelfxBillardGameRules(selfx.SelfxGameRules):
    def __init__(self, ctx):
        super(SelfxBillardGameRules, self).__init__(ctx)

    def on_world_setp(self, src, **pwargs):
        pass


class SelfxBillardAgent(selfx.SelfxAgent):
    def __init__(self, ctx):
        super(SelfxBillardAgent, self).__init__(ctx)


class SelfxBillardScope(selfx.SelfxScope):
    def __init__(self, ctx):
        super(SelfxBillardScope, self).__init__(ctx)

