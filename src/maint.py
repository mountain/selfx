# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import os
import argparse
import gym
import tianshou as ts
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from pathlib import Path
from gym import wrappers, logger
from tianshou.utils.net.discrete import Actor
from leibniz.nn.net import resnet


parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=256000, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g


logger.set_level(logger.INFO)
outdir = 'results/selfx-billard'
model_path = Path(outdir)

env = gym.make('selfx-billard-v0')
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
env.reset()


resize = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
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


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        h, w, a = state_shape[0], state_shape[1], action_shape
        self.output_dim = a
        self.resnet = resnet(9, a, layers=4, ratio=0,
            vblks=[2, 2, 2, 2], scales=[-2, -2, -2, -2],
            factors=[1, 1, 1, 1], spatial=(h, w))

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            obs = obs.cuda() if cuda else obs

        result = self.resnet(obs)
        return result, state


net = Actor(Net((screen_height, screen_width), 2 * n_actions), action_shape=[n_actions], hidden_sizes=[2 * n_actions])
if cuda:
    net = net.cuda()

optimizer = optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.RayVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(2)])
test_envs = ts.env.RayVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(2)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(total_size=10000, buffer_num=64))
test_collector = ts.data.Collector(policy, test_envs)


def save(mean_rewards):
    torch.save(policy.state_dict(), model_path / f'perf_{mean_rewards}.chk')
    return mean_rewards


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=1000, step_per_epoch=1000,
        episode_per_test=100, batch_size=8,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1), step_per_collect=100,
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
