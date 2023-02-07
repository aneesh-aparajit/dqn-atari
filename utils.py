import cv2
import gym
import numpy as np
import ipdb
from random import randint
import time


class FrameStackingEnv:
    def __init__(self, env: gym.Env, w: int, h: int, n: int = 4):
        '''FrameStackingEnv
        Inputs:
        ------
            - env: gym env
            - w: width of scaled image
            - h: height of scaled image
            - n: number of frames to be stacked.
        '''

        self.env = env
        self.w = w
        self.h = h
        self.n = n

        self.frames = np.zeros((n, h, w), np.uint8)
        self.frame = None

    def _preprocess_frame(self, frame):
        im = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (self.w, self.h))
        return im

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames[1:self.n, :, :] = self.frames[:self.n-1, :, :]
        self.frames[0, :, :] = self._preprocess_frame(state)
        self.frame = self._preprocess_frame(state)
        return self.frames.copy(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.frame = self._preprocess_frame(observation)
        self.frames = np.stack([self._preprocess_frame(observation)]*self.n)
        return self.frames.copy()

    def render(self, mode):
        self.env.render(mode)


if __name__ == '__main__':
    env = FrameStackingEnv(
        env=gym.make("Breakout-v4"),
        w=480,
        h=620
    )

    obs = env.reset()


    while True:
        action = randint(0, 3)
        s, r, d, i = env.step(action)

        if d:
            env.reset()

        env.render(mode="human")
        time.sleep(0.1)
    # ipdb.set_trace()
