# -*- coding: utf-8 -*-

import random
import numpy as np
import torch as th

import gym_selfx.selfx.selfx as selfx

from gym_selfx.render.draw import OpencvDrawFuncs
from Box2D.Box2D import (b2CircleShape as circleShape, b2World)


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS


class SelfxBillardToolkit(selfx.SelfxToolkit):
    def __init__(self):
        super(SelfxBillardToolkit, self).__init__()

    def build_inner_world(self, ctx):
        return SelfxBillardInnerWorld(ctx)

    def build_outer_world(self, ctx):
        return SelfxBillardOuterWorld(ctx)

    def build_game(self, ctx):
        return SelfxBillardGame(ctx)

    def build_agent(self, inner_world):
        return SelfxBillardAgent(inner_world)

    def build_scope(self, ctx):
        return SelfxBillardScope(ctx)


class SelfxBillardGame(selfx.SelfxGame):
    def __init__(self, ctx):
        super(SelfxBillardGame, self).__init__(ctx)

    def on_setpped(self, src, **pwargs):
        pass


class SelfxBillardWorld(selfx.SelfxWorld):
    def __init__(self, ctx, aname):
        super(SelfxBillardWorld, self).__init__(ctx, aname)

        self.x_threshold = 1029
        self.y_threshold = 645
        self.x_pos = self.x_threshold // 2
        self.y_pos = self.y_threshold // 2

        self.drawer = OpencvDrawFuncs(w=self.x_threshold, h=self.y_threshold, ppm=0.9)
        self.drawer.install()

        self.b2 = b2World(gravity=(0, 0), doSleep=True)

        self._state = self.available_states()[0]
        self._action = self.available_actions()[0]

    def available_actions(self):
        sacts = super(SelfxBillardWorld, self).availabe_actions()
        return sacts[:-1] + (
        ) + (sacts[-1],)

    def available_states(self):
        sstts = super(SelfxBillardWorld, self).availabe_states()
        return sstts[:-1] + (
        ) + (sstts[-1],)

    def action(self):
        return self._action

    def state(self):
        return self._state

    def act(self, actions):
        if self.ctx['agent'].b2.userData['energy'] <= 0:
            return selfx.QUIT
        else:
            return selfx.NOOP

    def reset(self):
        self.drawer.clear_screen()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            self.drawer.clear_screen()
            self.drawer.draw_world(self.b2)
            return self.drawer.screen
        else:
            return None

    def add_obstacle(self):
        if 30 < self.x_pos < self.x_threshold - 30 and 30 < self.y_pos < self.y_threshold - 30:
            self.b2.CreateStaticBody(
                position=(self.x_pos, self.y_pos),
                shapes=circleShape(radius=random.random() * 30),
                linearDamping=0.0,
                bullet=True,
                userData={
                    'world': self.b2,
                    'type': 'obstacle',
                    'ax': 0,
                    'ay': 0,
                    'color': (int(random.random() * 127) + 128, 128, int(random.random() * 127) + 128)
                },
            )

    def add_candy(self):
        self.b2.CreateDynamicBody(
            position=(self.x_pos, self.y_pos),
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

    def up(self):
        self.y_pos += 10
        self.y_pos = self.y_pos % self.y_threshold

    def dn(self):
        self.y_pos -= 10
        self.y_pos = self.y_pos % self.y_threshold

    def lf(self):
        self.x_pos -= 10
        self.x_pos = self.x_pos % self.x_threshold

    def rt(self):
        self.x_pos += 10
        self.x_pos = self.x_pos % self.x_threshold

    def random_walk(self, times):
        for _ in range(times):
            d = int(random.random() * 4)
            if d == 0:
                self.up()
            elif d == 1:
                self.dn()
            elif d == 2:
                self.lf()
            elif d == 3:
                self.rt()


class SelfxBillardInnerWorld(SelfxBillardWorld):
    def __init__(self, ctx):
        super(SelfxBillardInnerWorld, self).__init__(ctx, 'inner')

    def available_actions(self):
        sacts = super(SelfxBillardWorld, self).availabe_actions()
        return sacts[:-1] + (
            'up', 'dn', 'lf', 'rt',
            'add_obstacle'
        ) + (sacts[-1],)

    def step(self, action):
        game = self.ctx['game']
        if type(action) == th.Tensor:
            action = action.item()
            action = game.action_space()[action]
            action = action.inner
        if type(action) == list:
            action = action[0]

        if action == 'up':
            self.up()

        if action == 'dn':
            self.dn()

        if action == 'lf':
            self.lf()

        if action == 'rt':
            self.rt()

        if action == 'add_obstacle':
            self.add_obstacle()

        self.b2.Step(TIME_STEP, 10, 10)

        for b in self.b2.bodies:
            x, y = b.position
            b.position = x % self.x_threshold, y % self.y_threshold


class SelfxBillardOuterWorld(SelfxBillardWorld):
    def __init__(self, ctx):
        super(SelfxBillardOuterWorld, self).__init__(ctx, 'outer')
        for _ in range(30):
            self.random_walk(1000)
            self.add_obstacle()

    def step(self, action):
        game = self.ctx['game']
        if type(action) == th.Tensor:
            action = action.item()
            action = game.action_space()[action]
            action = action.outer
        if type(action) == list:
            action = action[0]

        if random.random() > 0.98 + 0.02 * np.exp(-len(self.b2.bodies)):
            self.random_walk(1000)
            self.add_candy()

        self.b2.Step(TIME_STEP, 10, 10)

        self.scores = self.ctx['agent'].b2.userData['energy']

        for b in self.b2.bodies:
            x, y = b.position
            b.position = x % self.x_threshold, y % self.y_threshold

        if action == selfx.NOOP:
            self.scores -= 1.0
            self.fire_step_event(action=action)

        if action == selfx.QUIT:
            self.status = selfx.OUT_GAME

        super(SelfxBillardOuterWorld, self).step(action=action)


class SelfxBillardAgentMouth(selfx.SelfxAffordable):
    def __init__(self, ctx):
        super(SelfxBillardAgentMouth, self).__init__(ctx, 'mouth')

    def available_actions(self):
        return ('open', 'close')

    def available_states(self):
        return ('closed', 'opened')

    def open(self):
        self._state = 'closed'

    def close(self):
        self._state = 'opened'

    def on_stepped(self, src, **pwargs):
        action = pwargs['action']

        if action.mouth == 'open':
            self.open()

        if action.mouth == 'close':
            self.close()


class SelfxBillardAgentGear(selfx.SelfxAffordable):
    def __init__(self, ctx):
        super(SelfxBillardAgentGear, self).__init__(ctx, 'gear')

    def available_actions(self):
        return ('gear0', 'gear1', 'gear2', 'gear3')

    def available_states(self):
        return ('gear0', 'gear1', 'gear2', 'gear3')

    def gear0(self):
        self._state = 'gear0'

    def gear1(self):
        self._state = 'gear1'

    def gear2(self):
        self._state = 'gear2'

    def gear3(self):
        self._state = 'gear3'

    def on_stepped(self, src, **pwargs):
        action = pwargs['action']

        if action.gear == 'gear0':
            self.gear0()

        if action.gear == 'gear1':
            self.gear1()

        if action.gear == 'gear2':
            self.gear2()

        if action.gear == 'gear3':
            self.gear3()


class SelfxBillardAgentBrake(selfx.SelfxAffordable):
    def __init__(self, ctx):
        super(SelfxBillardAgentBrake, self).__init__(ctx, 'brake')

    def available_actions(self):
        return ('up', 'dn')

    def available_states(self):
        return ('up', 'dn')

    def up(self):
        self._state = 'up'

    def down(self):
        self._state = 'dn'

    def on_stepped(self, src, **pwargs):
        action = pwargs['action']

        if action.brake == 'up':
            self.up()

        if action.brake == 'dn':
            self.down()


class SelfxBillardAgent(selfx.SelfxAgent):
    def __init__(self, ctx):
        super(SelfxBillardAgent, self).__init__(ctx)
        angle = random.random() * 360
        alpha = np.deg2rad(angle)
        self.b2 = ctx['outer'].b2.CreateDynamicBody(
            position=(513, 321),
            angle=alpha,
            linearVelocity=(np.random.normal() * 500, np.random.normal() * 500),
            linearDamping=0.01,
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

        self.mouth = SelfxBillardAgentMouth(self.ctx)
        self.gear = SelfxBillardAgentGear(self.ctx)
        self.brake = SelfxBillardAgentBrake(self.ctx)

    def subaffordables(self):
        return self.mouth, self.gear, self.brake, self.inner_world

    def available_actions(self):
        return ('idle')

    def available_states(self):
        return ('idle')

    def get_center(self):
        return self.b2.position

    def on_stepped(self, src, **pwargs):
        action = pwargs['action']
        game = self.ctx['game']
        if type(action) == th.Tensor:
            action = action.item()
            action = game.action_space()[action]
            action = action[src.name()]
        if type(action) == list:
            action = action[0]

        vx, vy = self.b2.linearVelocity
        vx = vx + self.b2.userData['ax'] * TIME_STEP
        vy = vy + self.b2.userData['ay'] * TIME_STEP
        self.b2.linearVelocity = vx, vy
        self.b2.userData['energy'] = self.b2.userData['energy'] - np.abs(
            (np.abs(self.b2.userData['ax'] * vx) + np.abs(self.b2.userData['ay'] * vy)) * TIME_STEP
        )

        for contact in self.b2.contacts:
            other = contact.other
            if other.userData['type'] == 'obstacle':
                self.b2.userData['energy'] = self.b2.userData['energy'] - 10
            elif other.userData['type'] == 'candy':
                self.b2.userData['energy'] = self.b2.userData['energy'] + 10 * other.mass
                del other


class SelfxBillardScope(selfx.SelfxScope):
    def __init__(self, ctx):
        super(SelfxBillardScope, self).__init__(ctx)

