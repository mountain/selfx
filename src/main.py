#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import gym

from gym import wrappers, logger


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('selfx-billard-v0')

    outdir = 'results/selfx-billard'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    game = env.game

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = game.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            env.render(mode='rgb_array')

    # Close the env and write monitor result info to disk
    env.close()
