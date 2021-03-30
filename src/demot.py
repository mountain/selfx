#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import gym
import argparse
import torch
import os
import numpy as np

from pathlib import Path
from gym import wrappers, logger

import torchvision.transforms as T
from tianshou.utils.net.discrete import Actor
from gym_selfx.nn.modelt import Net


logger.set_level(logger.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=10, help="number of demo examples")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
parser.add_argument("-m", type=str, default='', help="model path")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g

device = torch.device(int(opt.g) if torch.cuda.is_available() else "cpu")

env = gym.make('selfx-billard-v0')
outdir = 'demo/selfx-billard'
env = wrappers.Monitor(env, directory=outdir, force=True, mode='evaluation', video_callable=lambda episode_id: True)
env.seed(0)
env.reset()


resize = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0)
    return screen.cuda() if cuda else screen

init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
n_actions = len(env.action_space)

if opt.m == '':
    pattern = '*.chk'
else:
    pattern = opt.m

model_path = Path('results/selfx-billard')
policy_net = Actor(Net((screen_height, screen_width), 2 * n_actions, cuda), action_shape=[n_actions], hidden_sizes=[2 * n_actions], device=device)
policy_net.load_state_dict(torch.load(sorted(list(model_path.glob(pattern)))[-1], map_location=device)['policy'])


def select_action(observation, reward, done):
    with torch.no_grad():
        expected_reward = policy_net(observation)
        return expected_reward.max(1)[1].view(1, 1)


if __name__ == '__main__':
    episode_count = opt.n

    for i in range(episode_count):
        env.reset()
        env.game.policy = select_action
        reward = 0
        done = False

        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = torch.cat((current_screen, last_screen), dim=1)
        i = 0
        while True:
            i += 1

            action = env.game.act(state, reward, done)
            _, reward, done, info = env.step(action)

            last_screen = current_screen
            current_screen = get_screen(env, device)
            state = torch.cat((current_screen, last_screen), dim=1)

            if done or i > 54000:
                break
            env.render(mode='rgb_array')

    # Close the env and write monitor result info to disk
    env.close()
