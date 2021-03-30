# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import os
import argparse
import gym
import tianshou as ts
import numpy as np

import torch
import torch.optim as optim
import torchvision.transforms as T

from pathlib import Path
from gym import wrappers, logger

from tianshou.utils.net.discrete import Actor
from gym_selfx.nn.modelt import Net


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1024, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g
device = 'cuda' if cuda else 'cpu'


logger.set_level(logger.INFO)
outdir = 'results/selfx-billard'
model_path = Path(outdir)

env = gym.make('selfx-billard-v0')
env = wrappers.Monitor(env, directory=outdir, force=True)
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

net = Actor(Net((screen_height, screen_width), 2 * n_actions, cuda), action_shape=[n_actions], hidden_sizes=[2 * n_actions], device=device)
if cuda:
    net = net.cuda()

optimizer = optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.SubprocVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(16)])
test_envs = ts.env.SubprocVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(32)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(total_size=8192, buffer_num=64), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs)


def save(policy):
    test_collector.collect(n_episode=1, render=1 / 35)
    torch.save(policy.state_dict(), model_path / f'policy.chk')


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=8192, step_per_epoch=512,
        episode_per_test=128, batch_size=24,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1), step_per_collect=32,
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        save_fn=lambda policy: save(policy),
    )
