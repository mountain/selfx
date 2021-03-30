import torch as th
import torch.nn as nn

from leibniz.nn.net import resnet
from leibniz.nn.layer.hyperbolic import HyperBottleneck



class Net(nn.Module):
    def __init__(self, state_shape, action_shape, cuda):
        super().__init__()
        self.cuda = cuda
        h, w, a = state_shape[0] // 3, state_shape[1], action_shape
        self.output_dim = a
        self.resnet = resnet(9, a, layers=4, ratio=0, block=HyperBottleneck,
            vblks=[2, 2, 2, 2], scales=[-2, -2, -2, -2],
            factors=[1, 1, 1, 1], spatial=(h, w))
        if cuda:
            self.resnet.cuda()

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, th.Tensor):
            obs = th.tensor(obs, dtype=th.float)
            obs = obs.cuda() if self.cuda else obs

        result = self.resnet(obs)
        return result, state
