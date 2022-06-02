import cv2
import numpy as np

from gym import Wrapper, Env
from typing import Union, Tuple


class ActionRepeatWrapper(Wrapper):
    def __init__(self, env: Env, action_repeat: int):
        super().__init__(env)
        self._act_rep = action_repeat

    def step(self, action):
        """Steps the environment `action_repeat`-many times forward, terminating
        when done is hit early. Returns the last observed frame."""

        done = False
        tot_rew = 0

        for _ in range(self._act_rep):
            state, reward, done, info = self.env.step(action)
            tot_rew += reward

            if done:
                break

        return state, reward, done, info


class ImageWrapper(Wrapper):
    def __init__(self, env: Env, img_width=64, img_height=64, bit_depth=5):
        super().__init__(env)
        self.img_w = img_width
        self.img_h = img_height
        self._reduct = 1 << (8-bit_depth)

    def _get_image(self):
        state = self.env.render('rgb_array')
        state = cv2.resize(state, (self.img_w, self.img_h))

        state = state // self._reduct * self._reduct  # Reduce bit depth
        state = np.array(state / 255., dtype=np.float32)
        return state

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_image(), reward, done, info

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self._get_image()
