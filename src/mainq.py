#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import random
import numpy as np
import gym

from gym import wrappers, logger

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('selfx-billard-v0')

    outdir = 'results/selfx-billard'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = env.agent

    q_table = np.zeros([len(env.observation_space), len(env.action_space)])

    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[ob])  # Exploit learned values

            obx, reward, done, _ = env.step(action)
            if done:
                break

            oldval = q_table[ob, action]
            maxval = np.max(q_table[obx])

            newval = (1 - alpha) * oldval + alpha * (reward + gamma * maxval)
            q_table[ob, action] = newval
            ob = obx

            env.render(mode='rgb_array')

    # Close the env and write monitor result info to disk
    env.close()
