# -*- coding: utf-8 -*-

import random
import numpy as np
import torch as th
import itertools

import gym_selfx.selfx.selfx as selfx

from gym_selfx.render.draw import OpencvDrawFuncs
from Box2D.Box2D import (b2CircleShape as circleShape, b2World)
from collections import deque


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS

XTHRESHOLD = 1024
YTHRESHOLD = 512


class SelfxBillardToolkit(selfx.SelfxToolkit):
    def __init__(self):
        super(SelfxBillardToolkit, self).__init__()

    def build_inner_world(self, ctx):
        return SelfxBillardInnerWorld(ctx)

    def build_outer_world(self, ctx):
        return SelfxBillardOuterWorld(ctx)

    def build_game(self, ctx):
        return SelfxBillardGame(ctx)

    def build_agent(self, inner_world, eye):
        return SelfxBillardAgent(inner_world, eye)

    def build_eye(self, ctx):
        return SelfxBillardEye(ctx)

    def build_scope(self, ctx):
        return SelfxBillardScope(ctx)


class SelfxBillardGame(selfx.SelfxGame):
    def __init__(self, ctx):
        super(SelfxBillardGame, self).__init__(ctx)
        self.total = 0.0
        self.queue = deque(maxlen=5)

    def reward(self):
        energy = self.ctx['agent'].b2.userData['energy']
        self.total = self.total + energy
        return energy

    def reset(self):
        self.queue.append(self.total)
        self.total = 0.0

    def round_begin(self):
        pass

    def round_end(self):
        self.queue.clear()
        self.total = 0.0

    def exit_condition(self):
        return self.ctx['agent'].b2.userData['energy'] <= 500

    def force_condition(self):
        return random.random() < 1 / TARGET_FPS / 7

    def performance(self):
        return np.array(self.queue).mean()


class SelfxBillardWorld(selfx.SelfxWorld):
    def __init__(self, ctx, aname):
        super(SelfxBillardWorld, self).__init__(ctx, aname)

        self.x_threshold = XTHRESHOLD
        self.y_threshold = YTHRESHOLD
        self.x_pos = self.x_threshold // 2
        self.y_pos = self.y_threshold // 2

        self.drawer = OpencvDrawFuncs(w=self.x_threshold, h=self.y_threshold, ppm=1.0)
        self.b2 = b2World(gravity=(0, 0), doSleep=True)

        self._state = self.available_states()[0]
        self._action = self.available_actions()[0]

    def action(self):
        return self._action

    def state(self):
        return self._state

    def reset(self):
        self.drawer.clear_screen()

        for b in self.b2.bodies:
            self.b2.DestroyBody(b)

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
                shapes=circleShape(radius=20),
                linearDamping=0.0,
                bullet=True,
                userData={
                    'world': self.b2,
                    'type': 'obstacle',
                    'ax': 0,
                    'ay': 0,
                    'color': (192, 128, 128)
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
                'color': (128, 255, 128)
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


def get_action(ctx, src, **pwargs):
    action = pwargs['action']
    game = ctx['game']
    if type(action) == th.Tensor:
        action = action.item()
        action = game.action_space()[action]
    if type(action) == list:
        action = action[0]

    return action


class SelfxBillardInnerWorld(SelfxBillardWorld):
    def __init__(self, ctx):
        super(SelfxBillardInnerWorld, self).__init__(ctx, 'inner')

    def available_actions(self):
        return (
            'up', 'dn', 'lf', 'rt',
            'add_obstacle'
        )

    def step(self, **pwargs):
        super(SelfxBillardInnerWorld, self).step(pwargs)
        action = get_action(self.ctx, self, **pwargs)

        if action.inner == 'up':
            self.up()

        if action.inner == 'dn':
            self.dn()

        if action.inner == 'lf':
            self.lf()

        if action.inner == 'rt':
            self.rt()

        if action.inner == 'add_obstacle':
            selfx.debug = True
            self.add_obstacle()

        self.b2.Step(TIME_STEP, 10, 10)
        for b in self.b2.bodies:
            x, y = b.position
            b.position = x % self.x_threshold, y % self.y_threshold


class SelfxBillardOuterWorld(SelfxBillardWorld):
    def __init__(self, ctx):
        super(SelfxBillardOuterWorld, self).__init__(ctx, 'outer')
        self.reset()

    def reset(self):
        super(SelfxBillardOuterWorld, self).reset()
        for _ in range(10):
            self.random_walk(1000)
            self.add_obstacle()

    def step(self, **pwargs):
        super(SelfxBillardOuterWorld, self).step(**pwargs)
        action = get_action(self.ctx, self, **pwargs)
        self.fire_step_event(action=action)

        if random.random() > 1 - np.exp(-len(self.b2.bodies) / 30):
            self.random_walk(1000)
            self.add_candy()

        self.b2.Step(TIME_STEP, 10, 10)
        for b in self.b2.bodies:
            x, y = b.position
            b.position = x % self.x_threshold, y % self.y_threshold


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

    def reset(self):
        self._state = self.available_states()[0]

    def on_stepped(self, src, **pwargs):
        action = get_action(self.ctx, src, **pwargs)

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

    def value(self):
        return float(self._state[4:])

    def reset(self):
        self._state = self.available_states()[0]

    def on_stepped(self, src, **pwargs):
        action = get_action(self.ctx, src, **pwargs)

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

    def value(self):
        if self._state == 'up':
            return 0.0
        else:
            return 0.8

    def reset(self):
        self._state = self.available_states()[0]

    def on_stepped(self, src, **pwargs):
        action = get_action(self.ctx, src, **pwargs)

        if action.brake == 'up':
            self.up()

        if action.brake == 'dn':
            self.down()


class SelfxBillardAgentSteer(selfx.SelfxAffordable):
    def __init__(self, ctx):
        super(SelfxBillardAgentSteer, self).__init__(ctx, 'steer')

    def available_actions(self):
        return ('l2', 'l1', 'o', 'r1', 'r2')

    def available_states(self):
        return ('l2', 'l1', 'o', 'r1', 'r2')

    def l2(self):
        self._state = 'l2'

    def l1(self):
        self._state = 'l1'

    def o(self):
        self._state = 'o'

    def r2(self):
        self._state = 'r2'

    def r1(self):
        self._state = 'r1'

    def value(self):
        if self._state == 'o':
            return 0.0
        elif self._state == 'l2':
            return np.pi / 3
        elif self._state == 'l1':
            return np.pi / 6
        elif self._state == 'r2':
            return - np.pi / 3
        elif self._state == 'r1':
            return - np.pi / 6
        else:
            return 0.0

    def reset(self):
        self._state = self.available_states()[0]

    def on_stepped(self, src, **pwargs):
        action = get_action(self.ctx, src, **pwargs)

        if action.steer == 'o':
            self.o()

        if action.steer == 'l1':
            self.l1()

        if action.steer == 'l2':
            self.l2()

        if action.steer == 'r1':
            self.r1()

        if action.steer == 'r2':
            self.r2()


class SelfxBillardAgent(selfx.SelfxAgent):
    def __init__(self, ctx, eye):
        super(SelfxBillardAgent, self).__init__(ctx, eye)
        self.mouth = SelfxBillardAgentMouth(self.ctx)
        self.gear = SelfxBillardAgentGear(self.ctx)
        self.brake = SelfxBillardAgentBrake(self.ctx)
        self.steer = SelfxBillardAgentSteer(self.ctx)
        self.reset()

    def reset(self):
        super(SelfxBillardAgent, self).reset()
        angle = random.random() * 360
        alpha = np.deg2rad(angle)
        self.b2 = self.ctx['outer'].b2.CreateDynamicBody(
            position=(self.ctx['outer'].x_threshold / 2, self.ctx['outer'].y_threshold / 2),
            angle=alpha,
            linearVelocity=(np.random.normal() * 500, np.random.normal() * 500),
            linearDamping=0.0,
            bullet=True,
            userData= {
                'world': self.ctx['outer'].b2,
                'type': 'monster',
                'energy': 1000,
                'ax': 0,
                'ay': 0,
                'color': (255, 255, 0)
            }
        )
        self.b2.userData['energy'] = self.b2.userData['energy'] + 10 * self.b2.mass
        self.b2.CreateCircleFixture(radius=5.0, density=1, friction=0.0)

    def subaffordables(self):
        return self.mouth, self.gear, self.brake, self.steer, self.inner_world

    def available_actions(self):
        return ('idle',)

    def available_states(self):
        return ('idle',)

    def center(self):
        return self.b2.position

    def direction(self):
        return self.b2.linearVelocity

    def on_stepped(self, src, **pwargs):
        super(SelfxBillardAgent, self).on_stepped(src, **pwargs)

        gear_value = self.gear.value()
        brake_value = self.brake.value()
        steer_value = self.steer.value()

        vx, vy = self.b2.linearVelocity
        angle = np.arctan2(vy, vx)
        fx = gear_value * np.cos(angle + steer_value)
        fy = gear_value * np.sin(angle + steer_value)
        ax = (-0.01 * vx - 0.1 * brake_value * vx + fx) / self.b2.mass
        ay = (-0.01 * vy - 0.1 * brake_value * vy + fy) / self.b2.mass
        vx = vx + ax * TIME_STEP
        vy = vy + ay * TIME_STEP
        self.b2.linearVelocity = vx, vy
        energy_loss = (fx * vx + fy * vy) * TIME_STEP
        self.b2.userData['ax'] = ax
        self.b2.userData['ay'] = ay
        self.b2.userData['energy'] = self.b2.userData['energy'] - energy_loss
        self.b2.mass = self.b2.mass - energy_loss / 10

        mouth_open = self.ctx['game'].state().mouth == 'opened'
        if mouth_open:
            self.b2.userData['color'] = (255, 192, 0)
        else:
            self.b2.userData['color'] = (255, 255, 0)

        for contact in self.b2.contacts:
            other = contact.other
            if other.userData['type'] == 'obstacle':
                self.b2.userData['energy'] = self.b2.userData['energy'] - 10
                self.b2.mass = self.b2.mass - 1
            elif other.userData['type'] == 'candy':
                if mouth_open:
                    mass0 = self.b2.mass
                    mass1 = other.mass
                    self.b2.userData['energy'] = self.b2.userData['energy'] + 10 * mass1
                    self.b2.mass = mass0 + mass1

                    vx, vy = self.b2.linearVelocity
                    ux, uy = other.linearVelocity
                    wx = (vx * mass0 + ux * mass1) / self.b2.mass
                    wy = (vy * mass0 + uy * mass1) / self.b2.mass
                    self.b2.linearVelocity = wx, wy

                    self.ctx['outer'].b2.DestroyBody(other)


class SelfxBillardEye(selfx.SelfxEye):
    def __init__(self, ctx):
        super(SelfxBillardEye, self).__init__(ctx)
        self.x_threshold = XTHRESHOLD // 4
        self.y_threshold = YTHRESHOLD // 4

        self.drawer = OpencvDrawFuncs(w=self.x_threshold, h=self.y_threshold, ppm=1.0)
        self.b2 = b2World(gravity=(0, 0), doSleep=True)

    def reset(self):
        self.drawer.clear_screen()
        for b in self.b2.bodies:
            self.b2.DestroyBody(b)

        self.ownbody = self.b2.CreateStaticBody(
            position=(self.x_threshold / 2, self.y_threshold / 2),
            shapes=circleShape(radius=5.0),
            linearDamping=0.0,
            bullet=True,
            userData= {
                'color': (255, 255, 0)
            }
        )

    def view(self, world, center, direction):
        self.reset()

        x0, y0 = center
        rx, ry = direction
        theta = np.arctan2(ry, rx)
        rx, ry = np.cos(theta), np.sin(theta)
        px, py = np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)
        for b in world.b2.bodies:
            x, y = b.position
            dx, dy = x - x0, y - y0
            alngr = rx * dx + ry * dy
            alngp = px * dx + py * dy

            if - self.x_threshold / 2 < alngr < self.x_threshold / 2 and - self.y_threshold / 2 < alngp < self.y_threshold / 2:
                bx, by = self.ownbody.position
                if b.userData['type'] == 'candy':
                    self.b2.CreateStaticBody(
                        position=(bx - alngr, by - alngp),
                        shapes=circleShape(radius=5.0),
                        linearDamping=0.0,
                        bullet=True,
                        userData= {
                            'color': (128, 255, 128)
                        }
                    )
                if b.userData['type'] == 'obstacle':
                    self.b2.CreateStaticBody(
                        position=(bx - alngr, by - alngp),
                        shapes=circleShape(radius=20.0),
                        linearDamping=0.0,
                        bullet=True,
                        userData= {
                            'color': (192, 128, 128)
                        }
                    )

        self.drawer.install()
        self.drawer.draw_world(self.b2)

        return self.drawer.screen

    def available_states(self):
        return itertools.product(
            '01', '01', '01'
        )

    def state(self):
        outer = self.ctx['outer']
        agent = self.ctx['agent']
        view = self.view(outer, agent.center(), agent.direction())

        w, h, _ = view.shape
        fl = view[:w // 2, :h // 2, :]
        fr = view[w // 2:, :h // 2, :]
        b = view[:, h // 2:, :]
        fl = fl * (fl < 255)
        fr = fr * (fr < 255)
        b = b * (b < 255)

        fl = np.max(fl) > 128
        fr = np.max(fr) > 128
        b = np.max(b) > 128

        bits = '%d%d%d' % (fl, fr, b)
        return bits


class SelfxBillardScope(selfx.SelfxScope):
    def __init__(self, ctx):
        super(SelfxBillardScope, self).__init__(ctx)

