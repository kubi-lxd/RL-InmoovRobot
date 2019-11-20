
from stable_baselines import PPO2
import pybullet as p
import numpy as np
from gym import spaces
import gym

from environments.inmoov import inmoov

URDF_PATH = "/home/tete/work/SJTU/inmoov/robotics-rl-srl/urdf_robot/"

class InmoovGymEnv(gym.Env):
    def __init__(self, urdf_path=URDF_PATH, multi_view=False, seed=0):
        self.seed(seed)
        self.urdf_path = URDF_PATH

        self._observation = None

        self._inmoov = None
        self.reset()

    def reset(self):
        self._inmoov = inmoov.Inmoov(urdf_path=self.urdf_path, positional_control=True)

    def _reward(self):
        # TODO
        return

    def _termination(self):
        # TODO
        return

    def step(self, action):
        self._inmoov.apply_action_pos(action)
        self._observation = self.render(mode='rgb')
        reward = self._reward()
        done = self._termination()
        # self.obs[:], rewards, self.dones, infos
        return np.array(self._observation), reward, done, {}

    def render(self, mode='rgb'):
        # TODO
        return np.array([])

    def close(self):
        return