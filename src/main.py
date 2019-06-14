#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import gym_selfx.selfx.selfx as selfx
import gym

from gym import wrappers, logger


class Agent(selfx.SelfxAgent):
    def __init__(self, inner_world):
        super(Agent, self).__init__(inner_world)


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    env = gym.make('selfx-bounday-candy-v0')

    outdir = '/tmp/results-selfx-bounday-candy'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = Agent(env.inner)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        env.render()

    # Close the env and write monitor result info to disk
    env.close()
