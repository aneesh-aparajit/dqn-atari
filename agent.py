import gym
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from typing import Optionl, Any
from dataclasses import dataclass
from random import sample

from model import DQNModel
from utils import FrameStackingEnv



@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, max_len: int = 100_000):
        self.max_len = max_len
        self.buffer = [None for _ in range(max_len)]
        self.idx = 0

    def insert(self, item: Sarsd):
        self.buffer[self.idx % self.max_len] = item
        self.idx += 1

    def sample(self, sample_size: int = 32):
        assert sample_size > min(self.idx, self.max_len), "There are not enough elements to sample from."
        if self.idx < self.max_len:
            return sample(self.buffer[:self.idx], sample_size)
        return sample(self.buffer, sample_size)


def update_target_model(m: nn.Module, tgt: nn.Module):
    '''A function to update the target model'''
    tgt.load_state_dict(m.state_dict())
    print('[Target] model weights  updated...')


def train_step()
