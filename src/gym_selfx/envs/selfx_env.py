# -*- coding: utf-8 -*-

import os, subprocess, time, signal
import gym
import gym_selfx.selfx.selfx as selfx

from gym import error, spaces
from gym import utils


import logging
logger = logging.getLogger(__name__)


class SelfXEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None

        self.configure()
        self.start_server()
        self.start_viewer()

        self.env = self.init_environment()
        self.env.connectToServer(self.server_port)

        self.env.outer = self.env.build_world()
        self.env.inner = self.env.build_world()

        self.status = selfx.IN_GAME

    def __del__(self):
        self.env.act(selfx.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def configure(self):
        pass

    def init_environment(self):
        return None

    def start_server(self):
        cmd = ''
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def start_viewer(self):
        cmd = ''
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def step(self, action):
        self.take_action(action)
        self.status = self.env.step()
        reward = self.get_reward()
        ob = self.env.getState()
        episode_over = self.status != selfx.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == selfx.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == selfx.TURN:
            self.env.act(action_type, action[3])
        elif action_type == selfx.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(selfx.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == selfx.GOAL:
            return 1
        else:
            return 0

    def reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == selfx.IN_GAME:
            self.env.act(selfx.NOOP)
            self.status = self.env.step()
        while self.status != selfx.IN_GAME:
            self.env.act(selfx.NOOP)
            self.status = self.env.step()
        return self.env.getState()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()
