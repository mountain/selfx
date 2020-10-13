# -*- coding: utf-8 -*-

# bootstrap
import gym_selfx.selfx

import os
import argparse
import gym
import tianshou as ts

import torch
import torch.optim as optim

from pathlib import Path
from gym import wrappers, logger
from gym_selfx.nn.dqn import get_screen
from tianshou.utils.net.discrete import DQN

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=256000, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g

device = torch.device(int(opt.g) if torch.cuda.is_available() else "cpu")
logger.set_level(logger.INFO)
outdir = 'results/selfx-billard'
model_path = Path(outdir)

env = gym.make('selfx-billard-v0')
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
env.reset()

init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape
n_actions = len(env.action_space)

env = None

net = DQN(3, screen_height, screen_width, n_actions, device=device).to(device)
optimizer = optim.Adam(net.parameters())
policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_envs = ts.env.DummyVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(8)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('selfx-billard-v0') for _ in range(100)])

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)

if __name__ == '__main__':
    logger.set_level(logger.INFO)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=1000, collect_per_step=10,
        episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        writer=None)

    print(f'training | duration: {result["duration"]}, best: {result["best_reward"]}')

    perf = env.game.performance()
    dura = env.game.avg_duration()

    filepath = model_path / f'perf_{int(perf):010d}.duration_{int(dura):04d}.chk'
    torch.save(policy.state_dict(), filepath)
