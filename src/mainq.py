# -*- coding: utf-8 -*-

"""
This code is originally derived from
 Reinforcement Learning (DQN) Tutorial by `Adam Paszke <https://github.com/apaszke>`

"""

# bootstrap
import gym_selfx.selfx

import os
import argparse
import gym
import math
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from itertools import count
from gym import wrappers, logger
from dqn import DQN, ReplayMemory, Transition, get_screen


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1000, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.set_level(logger.INFO)

env = gym.make('selfx-billard-v0')
outdir = 'results/selfx-billard'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
env.reset()

init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape
n_actions = len(env.action_space)

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

steps_done = 0
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(policy_net.parameters())


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            expected_reward = policy_net(state)
            return expected_reward.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.long)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    action_batch = [actions for actions in batch.action]
    action_batch = torch.cat(action_batch)

    reward_batch = batch.reward
    reward_batch = torch.cat(reward_batch)

    state_batch = batch.state
    state_batch = torch.cat(state_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask == 1] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    env = gym.make('selfx-billard-v0')

    outdir = 'results/selfx-billard'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    game = env.game

    ## code in main.py for comparison
    #
    # reward = 0
    # done = False
    # for i in range(opt.n):
    #     ob = env.reset()
    #
    #     while True:
    #         action = game.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         if done:
    #             break
    #         env.render(mode='rgb_array')

    for i_episode in range(opt.n):
        env.reset()

        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = torch.cat((current_screen, last_screen), dim=1)

        for t in count():
            action = select_action(state)
            _, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = torch.cat((current_screen, last_screen), dim=1)
            else:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()
            if done:
                print(f'duration[{i_episode:04d}]:{t + 1:04d}')
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            model_path = Path(outdir)
            torch.save(policy_net.state_dict(), model_path / f'duration_{t + 1:04d}.episode_{i_episode:04d}.mdl')
            glb = list(model_path.glob('*.mdl'))
            if len(glb) > 20:
                for p in sorted(glb)[:-15]:
                    os.unlink(p)

    env.close()
