#!/usr/bin/env python
# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import gym
import argparse
import os
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from gym import wrappers, logger

import tianshou as ts
import torchvision.transforms as T

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
    T.Resize((256 * 3, 512), interpolation=Image.CUBIC),
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
net = Net((screen_height, screen_width), 2 * n_actions, cuda)
policy_net = ts.policy.DQNPolicy(net, None, discount_factor=0.9, estimation_step=3, target_update_freq=320)
policy_net.load_state_dict(torch.load(sorted(list(model_path.glob(pattern)))[-1], map_location=device))


def select_action(observation, reward, done):
    with torch.no_grad():
        if torch.is_tensor(observation):
            b, _, h, w = observation.size()
            r, g, b = observation[:, 0:1], observation[:, 1:2], observation[:, 2:3]
            r1, r2, r3 = r[:, :, :h // 3], r[:, :, h // 3:2 * h // 3], r[:, :, 2 * h // 3:]
            g1, g2, g3 = g[:, :, :h // 3], g[:, :, h // 3:2 * h // 3], g[:, :, 2 * h // 3:]
            b1, b2, b3 = b[:, :, :h // 3], b[:, :, h // 3:2 * h // 3], b[:, :, 2 * h // 3:]
            observation = torch.cat((r1, g1, b1, r2, g2, b2, r3, g3, b3), dim=1)
        else:
            c, h, w = observation.shape
            observation = np.reshape(observation, (1, c, h, w))

        batch = ts.data.Batch({'obs': observation, 'info': {}})
        expected_reward = policy_net(batch).logits[0].view(2, 400)[0]
        index = torch.argmax(expected_reward, dim=0).item()
        action = env.action_space[index]
        return [action]


if __name__ == '__main__':
    episode_count = opt.n

    for i in range(episode_count):
        env.reset()
        env.game.policy = select_action
        reward = 0
        done = False

        i = 0
        state = env.state()
        while True:
            i += 1

            action = env.game.act(state, reward, done)
            _, reward, done, info = env.step(action)

            state = env.state()

            if done or i > 54000:
                break
            env.render(mode='rgb_array')

    # Close the env and write monitor result info to disk
    env.close()
