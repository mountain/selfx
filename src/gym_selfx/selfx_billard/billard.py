# -*- coding: utf-8 -*-

import random
import numpy as np

import gym_selfx.selfx.selfx as selfx

from gym_selfx.render.draw import OpencvDrawFuncs
from Box2D.Box2D import (b2CircleShape as circleShape, b2World)


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
        self.x_threshold = 1029
        self.y_threshold = 645

        self.drawer = OpencvDrawFuncs(w=self.x_threshold, h=self.y_threshold, ppm=0.9)
        self.drawer.install()

        self.b2 = b2World(gravity=(0, 0), doSleep=True)
        for _ in range(30):
            self.b2.CreateStaticBody(
                position=(30 + random.random() * 990, 30 + random.random() * 600),
                shapes=circleShape(radius=random.random() * 30),
                linearDamping=0.0,
                bullet=True,
                userData= {
                    'world': self.b2,
                    'type': 'obstacle',
                    'ax': 0,
                    'ay': 0,
                    'color': (int(random.random() * 127) + 128, 128, int(random.random() * 127) + 128)
                },
            )

    def act(self, actions):
        if self.ctx['agent'].b2.userData['energy'] <= 0:
            return selfx.QUIT
        else:
            return selfx.NOOP

    def step(self, action):
        if random.random() > 0.98 + 0.02 * np.exp(-len(self.b2.bodies)):
            self.b2.CreateDynamicBody(
                position=(random.random() * 1025, random.random() * 641),
                linearVelocity=(np.random.normal() * 500 + 500, np.random.normal() * 500),
                angle=random.random() * 360,
                linearDamping=0.0,
                bullet=True,
                userData={
                    'world': self.b2,
                    'type': 'candy',
                    'ax': 0,
                    'ay': 0,
                    'color': (128, int(random.random() * 255), int(random.random() * 255))
                },
            ).CreateCircleFixture(radius=5.0, density=1, friction=0.0)

        self.b2.Step(TIME_STEP, 10, 10)

        for b in self.b2.bodies:
            x, y = b.position
            b.position = x % self.x_threshold, y % self.y_threshold

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
            return self.drawer.screen
        else:
            return None


class SelfxBillardGameRules(selfx.SelfxGameRules):
    def __init__(self, ctx):
        super(SelfxBillardGameRules, self).__init__(ctx)

    def on_world_setpped(self, src, **pwargs):
        pass


class SelfxBillardAgent(selfx.SelfxAgent):
    def __init__(self, ctx):
        super(SelfxBillardAgent, self).__init__(ctx)

        angle = random.random() * 360
        alpha = np.deg2rad(angle)
        self.b2 = ctx['outer'].b2.CreateDynamicBody(
            position=(513, 321),
            angle=alpha,
            linearVelocity=(np.random.normal() * 500, np.random.normal() * 500),
            linearDamping=0.0,
            bullet=True,
            userData= {
                'world': ctx['outer'].b2,
                'type': 'monster',
                'energy': 100,
                'ax': 0,
                'ay': 0,
                'color': (255, 255, 0)
            }
        )
        self.b2.CreateCircleFixture(radius=5.0, density=1, friction=0.0)

    def get_center(self):
        return self.b2.position

    def on_world_stepped(self, **pwargs):
        vx, vy = self.b2.linearVelocity
        vx = vx + self.b2.userData['ax'] * TIME_STEP
        vy = vy + self.b2.userData['ay'] * TIME_STEP
        self.b2.linearVelocity = vx, vy
        self.b2.userData['energy'] = self.b2.userData['energy'] - np.abs(
            (np.abs(self.b2.userData['ax'] * vx) + np.abs(self.b2.userData['ay'] * vy)) * TIME_STEP
        )

        for contact in self.b2.contacts:
            other = contact.other
            if other.userData['type'] == 'candy':
                self.b2.userData['energy'] = self.b2.userData['energy'] + 10 * other.mass
                del other


class SelfxBillardScope(selfx.SelfxScope):
    def __init__(self, ctx):
        super(SelfxBillardScope, self).__init__(ctx)

