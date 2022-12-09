import random

import torch as th
import torch.nn as nn

from torch.nn.functional import linear
from leibniz.nn.net import resnetz


class Recurrent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.nn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, th.Tensor):
            s = th.tensor(s, dtype=th.float)
        # s [bsz, len, dim] (training)
        # or [bsz, dim] (evaluation)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s.view([bsz * length, dim]))
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data with [bsz, len, ...]
            # but pyth rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (
                state['h'].transpose(0, 1).contiguous(),
                state['c'].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, -1])
        # make sure the 0-dim is batch size: [bsz, len, ...]
        return s, {'h': h.transpose(0, 1).detach(),
                   'c': c.transpose(0, 1).detach()}


class Net(nn.Module):
    def __init__(self, state_shape, action_shape, cuda):
        super().__init__()
        self.flag = cuda
        h, w, a = state_shape[0] // 3, state_shape[1], action_shape
        self.output_dim = a
        self.max_action_num = a
        self.resnet = resnetz(18, 2 * a, layers=4, spatial=(h, w))
        self.fc = nn.Linear(h * w * a, a)
        self.recrr = Recurrent(2 * a, a, 4 * a)

        self.co1 = th.scalar_tensor(2 * random.random() - 1, dtype=th.float32)
        self.co2 = th.scalar_tensor(2 * random.random() - 1, dtype=th.float32)
        self.co3 = th.scalar_tensor(2 * random.random() - 1, dtype=th.float32)
        self.co4 = th.scalar_tensor(2 * random.random() - 1, dtype=th.float32)

    def crossover(self, another):
        coeff = th.sigmoid(self.co1 + another.co1)
        self.resnet.enconvs[0].transform.conv.depthwise.weight = th.nn.Parameter(self.enconvs[0].transform.conv.depthwise.weight * coeff + another.enconvs[0].transform.conv.depthwise.weight * (1 - coeff))
        self.resnet.enconvs[0].transform.conv.depthwise.bias = th.nn.Parameter(self.enconvs[0].transform.conv.depthwise.bias * coeff + another.enconvs[0].transform.conv.depthwise.bias * (1 - coeff))
        coeff = th.sigmoid(self.co2 + another.co2)
        self.resnet.enconvs[1].transform.conv.depthwise.weight = th.nn.Parameter(self.enconvs[1].transform.conv.depthwise.weight * coeff + another.enconvs[1].transform.conv.depthwise.weight * (1 - coeff))
        self.resnet.enconvs[1].transform.conv.depthwise.bias = th.nn.Parameter(self.enconvs[1].transform.conv.depthwise.bias * coeff + another.enconvs[1].transform.conv.depthwise.bias * (1 - coeff))
        coeff = th.sigmoid(self.co3 + another.co3)
        self.resnet.enconvs[2].transform.conv.depthwise.weight = th.nn.Parameter(self.enconvs[2].transform.conv.depthwise.weight * coeff + another.enconvs[2].transform.conv.depthwise.weight * (1 - coeff))
        self.resnet.enconvs[2].transform.conv.depthwise.bias = th.nn.Parameter(self.enconvs[2].transform.conv.depthwise.bias * coeff + another.enconvs[2].transform.conv.depthwise.bias * (1 - coeff))
        coeff = th.sigmoid(self.co4 + another.co4)
        self.resnet.enconvs[3].transform.conv.depthwise.weight = th.nn.Parameter(self.enconvs[3].transform.conv.depthwise.weight * coeff + another.enconvs[3].transform.conv.depthwise.weight * (1 - coeff))
        self.resnet.enconvs[3].transform.conv.depthwise.bias = th.nn.Parameter(self.enconvs[3].transform.conv.depthwise.bias * coeff + another.enconvs[3].transform.conv.depthwise.bias * (1 - coeff))

        if random.random() > 0.90:
            self.co1 = th.tanh(self.co1 + another.co1) * random.random()
            self.co2 = th.tanh(self.co2 + another.co2) * random.random()
            self.co3 = th.tanh(self.co3 + another.co3) * random.random()
            self.co4 = th.tanh(self.co4 + another.co4) * random.random()

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, th.Tensor):
            obs = th.tensor(obs, dtype=th.float)
            obs = obs.cuda() if self.flag else obs

        result = self.fc(self.resnet(obs))
        result = self.recrr(result, state=state)

        return result
