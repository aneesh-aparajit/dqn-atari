import gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from typing import Optional, Any
from dataclasses import dataclass
from random import sample
from tqdm import tqdm
from random import random

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
        # print(f'{self.idx}, {self.max_len}, {sample_size}, {min(self.idx, self.max_len) < sample_size}')
        # ipdb.set_trace()
        assert sample_size < min(self.idx, self.max_len)
        if self.idx < self.max_len:
            return sample(self.buffer[:self.idx], sample_size)
        return sample(self.buffer, sample_size)


def update_target_model(m: nn.Module, tgt: nn.Module):
    '''A function to update the target model'''
    tgt.load_state_dict(m.state_dict())
    print('[Target] model weights  updated...')


def train_step(model, tgt, state_transitions, n_actions, device, gamma=0.99):
    curr_states = torch.stack([torch.tensor(s.state, dtype=torch.float32) for s in state_transitions]).to(device)
    rewards = torch.stack([torch.tensor(s.reward, dtype=torch.float32) for s in state_transitions]).to(device)
    mask = torch.stack([torch.tensor(0, dtype=torch.float32) if s.done else torch.tensor(1, dtype=torch.float32) for s in state_transitions]).to(device)
    next_states = torch.stack([torch.tensor(s.state, dtype=torch.float32) for s in state_transitions]).to(device)
    actions = torch.LongTensor([s.action for s in state_transitions]).to(device)


    model.optimizer.zero_grad()
    with torch.no_grad():
        q_values_next = tgt.forward(next_states).max(-1)[0]

    q_values_curr = model.forward(curr_states)

    # We have to calculate the loss functions for only those Q values, corresponding to the action we take.
    # So, to do this, we take the one hot encoded version and then calculate them.

    loss = ((rewards + gamma*q_values_next*mask - torch.sum(F.one_hot(actions)*q_values_curr, axis=-1))**2).mean()

    # ipdb.set_trace()

    loss.backward()

    model.optimizer.step()
    model.scheduler.step()

    return loss

def test_step(model, device):
    env_test = gym.make("Breakout-v4")
    env_test = FrameStackingEnv(env=env_test, w=84, h=84, n=4)

    obs = env_test.reset()
    done = False

    frames = [env_test.frame]
    game_steps = 0
    score = 0

    while not done and game_steps < 1000:
        action = model.forward(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)).argmax(-1).item()
        obs, reward, done, _ = env_test.step(action)
        frames.append(env_test.frame)
        score += reward
        game_steps += 1

    return score, np.stack(frames, 0)


def main():

    wandb.init(project="dqn-breakout", name="iteration-1")

    env = gym.make("Breakout-v4")
    env = FrameStackingEnv(env=env, w=84, h=84, n=4)

    device = torch.device("cuda" if torch.has_cuda else "cpu")

    hparams = {
        'sample_size': 32,
        'rb_size': 100_000,
        'min_rb_size': 300,
        'learning_rate': 3e-4,
        'steps_to_train': 128,
        'steps_to_update': 256,
        'steps_until_train': 0,
        'steps_until_update': 0,
        'gamma': 0.999,
        'step_num': 0,
        'eps': 1,
        'eps_min': 0.001,
        'eps_decay': 0.99995,
        'test_every': 3000,
    }

    pbar = tqdm()

    m = DQNModel(obs_shape=env.env.observation_space.shape, n_actions=env.env.action_space.n, device=device)
    tgt = DQNModel(obs_shape=env.env.observation_space.shape, n_actions=env.env.action_space.n, device=device)
    update_target_model(m, tgt)

    replay_buffer = ReplayBuffer()
    step_num = 0
    test_until = 0

    last_obs = env.reset()

    episode_rewards = []
    rolling_rewards = 0

    while True:
        pbar.update(1)
        eps = hparams['eps_decay'] ** step_num

        if random() < eps:
            action = env.env.action_space.sample()
        else:
            action = m(torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0).to(device)).argmax(-1).item()

        state, reward, done, _ = env.step(action)
        replay_buffer.insert(Sarsd(state=last_obs, action=action, reward=reward, next_state=state, done=done))
        rolling_rewards += reward
        last_obs = state

        if done:
            last_obs = env.reset()
            episode_rewards.append(rolling_rewards)
            rolling_rewards = 0

        step_num += 1
        hparams['steps_until_train'] += 1
        hparams['steps_until_update'] += 1
        test_until += 1

        if replay_buffer.idx >= hparams['min_rb_size'] and hparams['steps_until_train'] >= hparams['steps_to_train']:
            state_transitions = replay_buffer.sample(hparams['sample_size'])
            loss = train_step(m, tgt, state_transitions, env.env.action_space.n, device, hparams['gamma'])
            wandb.log({
                'loss': loss.item(),
                'epsilon': eps,
                'avg_rewards': np.mean(episode_rewards),
                'rolling_rewards': rolling_rewards,
            }, step=step_num)
            hparams['steps_until_train'] = 0

        if hparams['steps_until_update'] >= hparams['steps_to_update']:
            update_target_model(m, tgt)
            hparams['steps_until_update'] = 0


        if test_until >= hparams['test_every']:
            test_rewards, frames = test_step(m, device)

            wandb.log({
                'test_rewards': test_rewards,
                'test_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(test_rewards), fps=25, format='mp4')
            }, step=step_num)
            test_step = 0


    # ipdb.set_trace()


if __name__ == '__main__':
    main()
