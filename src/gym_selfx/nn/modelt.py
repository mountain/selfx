import torch as th
import torch.nn as nn

from leibniz.nn.net import resnet
from leibniz.nn.layer.hyperbolic import HyperBottleneck


class Recurrent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.nn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                          num_layers=3, batch_first=True)
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
            # but pytorch rnn needs [len, bsz, ...]
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
        self.resnet = resnet(18, 2 * a, layers=4, ratio=-2, block=HyperBottleneck,
            vblks=[1, 1, 1, 1], scales=[-2, -2, -2, -2], factors=[1, 1, 1, 1], spatial=(h, w))
        self.recrr = Recurrent(2 * a, a, 4 * a)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, th.Tensor):
            obs = th.tensor(obs, dtype=th.float)
            obs = obs.cuda() if self.flag else obs

        result = self.recrr(self.resnet(obs), state=state)

        return result
