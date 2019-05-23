# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import math
import scipy.signal
from collections import deque
from ..args_provider import ArgsProvider

class GAE:
    def __init__(self):
        ''' Initialization discounted_reward.
        Accepted arguments:
        ``discount``: discount factor of reward.'''
        self.args = ArgsProvider(
            call_from = self,
            define_args = [
                ("GAMMA", dict(type=float, default=0.99)),
                ("LAMBDA", dict(type=float, default=0.95)),
            ],
            more_args = ["value_node"],
        )
        self.buffer_r = deque()
        self.buffer_v = deque()
        self.buffer_terminal = deque()
        self.buffer_terminal.append(0)
        
    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def setR(self, R, stats):
        ''' Set rewards and feed to stats'''
        self.buffer_v.appendleft(R.cpu().detach().numpy())
        stats["init_reward"].feed(R.mean())

    def feed(self, batch, stats):
        '''
        Update discounted reward and feed to stats.

        Keys in a batch:

        ``r`` (tensor): immediate reward.

        ``terminal`` (tensor): whether the current game has terminated.

        Feed to stats: immediate reward and accumulated reward
        '''
        r = batch["r"]
        term = batch["terminal"]
        v = batch["V"]
        
        self.buffer_r.appendleft(r.cpu().detach().numpy())
        self.buffer_v.appendleft(v.cpu().detach().numpy())
        self.buffer_terminal.appendleft(term.cpu().detach().numpy())

        advantage = self.discount(
            np.asarray(self.buffer_r) + self.args.GAMMA * np.asarray(self.buffer_v)[1:] * (1-np.asarray(self.buffer_terminal)[1:])\
            -np.asarray(self.buffer_v)[:-1], self.args.GAMMA*self.args.LAMBDA
        )

        return torch.tensor(advantage + np.asarray(self.buffer_v)[:-1])
