import os
from stable_baselines import PPO2
import pybullet as p
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from ipdb import set_trace as tt
from environments.inmoov import inmoov
GRAVITY = -9.8
URDF_PATH = "/home/tete/work/SJTU/inmoov/robotics-rl-srl/urdf_robot/"

class InmoovGymEnv(gym.Env):
    def __init__(self, urdf_path=URDF_PATH, multi_view=False, seed=0, debug_mode=False):
        self.seed(seed)
        self.urdf_path = URDF_PATH

        self._observation = None
        self.debug_mode = debug_mode
        self._inmoov = None

        if debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.reset()

    def reset(self):
        self.terminated = False
        self.n_contacts = 0
        self.n_steps_outside = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF(os.path.join("/home/tete/work/SJTU/inmoov/robotics-rl-srl/pybullet_data", "plane.urdf"), [0, 0, 0])
        p.setGravity(0, 0, -10)

        self._inmoov = inmoov.Inmoov(urdf_path=self.urdf_path, positional_control=True)
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)


    def _reward(self):
        # TODO
        return

    def _termination(self):
        # TODO
        return

    def effector_position(self):
        return self._inmoov.getGroundTruth()

    def step(self, action):
        self._inmoov.apply_action_pos(action)
        p.stepSimulation()
        # printGreen(action)
        # printYellow(self._inmoov.getGroundTruth())
        # tt()
        # self._observation = self.render(mode='rgb')
        # reward = self._reward()
        # done = self._termination()
        # self.obs[:], rewards, self.dones, infos
        # return np.array(self._observation), reward, done, {}

    def render(self, mode='rgb'):
        # TODO
        return np.array([])

    def close(self):
        return