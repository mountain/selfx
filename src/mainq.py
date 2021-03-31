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
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from itertools import count
from gym import wrappers, logger
from gym_selfx.nn.dqn import ReplayMemory, Transition, get_screen
from gym_selfx.nn.modelt import Net

import redis

r = redis.Redis(host='localhost', port=6379, db=7, decode_responses=True)

if r.exists('selfx:prob:crossover') <= 0:
    r.set('selfx:prob:crossover', '%0.8f' % 0.33333333)


BATCH_SIZE = 8
STATE_SIZE = 400
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 400 * 640
TARGET_UPDATE = 7
ROUND_UPDATE = 21

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=256000, help="number of epochs of training")
parser.add_argument("-g", type=str, default='0', help="index of gpu")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.g

device = torch.device(int(opt.g) if torch.cuda.is_available() else "cpu")
logger.set_level(logger.INFO)

env = gym.make('selfx-billard-v0')
outdir = 'results/selfx-billard'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
env.reset()

init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape
n_actions = len(env.action_space)

policy_net = Net([screen_height, screen_width], n_actions, cuda).to(device)
target_net = Net([screen_height, screen_width], n_actions, cuda).to(device)
loader_net = Net([screen_height, screen_width], n_actions, cuda).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

steps_done = 0
memory = ReplayMemory(10000)
optimizer = optim.Adam(policy_net.parameters())


def nature_selection():
    global memory
    model_path = Path(outdir)

    population = sorted(list(model_path.glob("*.chk")))
    idx = int(random.random() * (len(population) - 1))
    file_base = population[idx]
    checkpoint = torch.load(file_base, map_location=device)
    policy_net.load_state_dict(checkpoint['policy'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    memory = checkpoint['memory']

    prob = float(r.get('selfx:prob:crossover'))
    avgrank = np.array([float(rank) for rank in r.lrange('selfx:ranks', 0, 45)]).mean()
    if avgrank < 0.3:
        prob = np.tanh(prob / 3 / avgrank)
        r.set('selfx:prob:crossover', '%0.8f' % prob)
    else:
        prob = prob * 0.99
        r.set('selfx:prob:crossover', '%0.8f' % prob)
    print('crossover prob:', prob)

    if idx < (len(population) - 1) * prob:
        file_another = random.sample(sorted(list(model_path.glob("*.chk"))), 1)[0]
        checkpoint = torch.load(file_another, map_location=device)
        loader_net.load_state_dict(checkpoint['policy'])
        policy_net.crossover(loader_net)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


def getdata(tensor):
    b, c, h, w = tensor.size()
    if c == 18:
        return tensor
    else:
        h = h // 3
        r = tensor[:, :, 0:1].view(1, 3 * h, w)
        g = tensor[:, :, 1:2].view(1, 3 * h, w)
        b = tensor[:, :, 2:3].view(1, 3 * h, w)
        r1, r2, r3 = r[:, :h, :], r[:, h:2 * h, :], r[:, 2 * h:, :]
        g1, g2, g3 = g[:, :h, :], g[:, h:2 * h, :], g[:, 2 * h:, :]
        b1, b2, b3 = b[:, :h, :], b[:, h:2 * h, :], b[:, 2 * h:, :]
        return torch.cat((r1, g1, b1, r2, g2, b2, r3, g3, b3), dim=1)


def select_action(observation, reward, done):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            if not torch.is_tensor(observation):
                c, h, w = observation.shape
                observation = torch.tensor(observation, device=device, dtype=torch.float).view(1, c, h, w)
            observation = getdata(observation)

            expected_reward = policy_net(observation)
            index = expected_reward[0].argmax(dim=1)
    else:
        index = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    return [env.action_space[index.item()]]


def optimize_model():
    policy_net.train()

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.long)
    non_final_next_states = [torch.unsqueeze(torch.tensor(state, device=device, dtype=torch.float), 0) for state in batch.next_state]
    non_final_next_states = torch.cat(non_final_next_states, dim=0)

    # space = env.action_space
    # action_batch = [torch.tensor([space.index(a) for a in actions], device=device, dtype=torch.long) for actions in batch.action]
    # action_batch = torch.cat(action_batch, dim=0)

    reward_batch = batch.reward
    reward_batch = torch.cat(reward_batch, dim=0)

    state_batch = [torch.unsqueeze(torch.tensor(state, device=device, dtype=torch.float), 0) for state in batch.state]
    state_batch = torch.cat(state_batch, dim=0)

    state_action_values = policy_net(state_batch)[0]
    next_state_values = torch.unsqueeze(target_net(non_final_next_states)[0], 1)
    next_state_values = next_state_values * (non_final_mask == 1).view(-1, 1, 1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.view(-1, 1, 1)

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
    game.policy = select_action

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

    env.reset()
    for i_episode in range(opt.n):
        try:
            reward = 0
            done = False

            if i_episode % ROUND_UPDATE == 0:
                game.round_begin()

                if i_episode != 0:
                    nature_selection()

            state = env.state()

            for t in count():
                action = game.act(state, reward, done)
                if not done:
                    _, reward, done, _ = env.step(action)
                    reward = torch.tensor([reward], device=device, dtype=torch.float)
                    next_state = env.state()
                else:
                    next_state = None

                memory.push(state, action, next_state, reward)
                state = next_state

                optimize_model()
                if done:
                    print(f'duration[{i_episode:04d}]:{t + 1:04d}')
                    env.reset()
                    break

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                model_path = Path(outdir)
                perf = env.game.performance()
                dura = env.game.avg_duration()
                check = {
                    'policy': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'memory': memory,
                }
                co1 = policy_net.co1.item()
                co2 = policy_net.co2.item()
                co3 = policy_net.co3.item()

                filepath = model_path / f'perf_{int(perf):010d}.duration_{int(dura):04d}.episode_{i_episode:04d}.co_{co1:0.4f}_{co2:0.4f}_{co3:0.4f}.chk'
                torch.save(check, filepath)
                try:
                    rank = sorted(list(model_path.glob("*.chk"))).index(filepath) / 45
                except ValueError:
                    rank = 0.0
                r.lpush('selfx:ranks', '%0.8f' % rank)
                r.ltrim('selfx:ranks', 0, 45)
                print('rank:', rank)

                glb = list(model_path.glob('*.chk'))
                if len(glb) > 45:
                    for p in sorted(glb)[:-45]:
                        os.unlink(p)

            if i_episode % ROUND_UPDATE == 0:
                if i_episode != 0:
                    game.round_end()

        except Exception as e:
            import traceback, sys
            traceback.print_exception(*sys.exc_info())
            print('error:', e)

    env.close()
