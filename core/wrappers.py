import gym

from gym import Wrapper, Env


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
