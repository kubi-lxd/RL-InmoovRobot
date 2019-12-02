import os
from stable_baselines import PPO2
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from ipdb import set_trace as tt
from environments.inmoov import inmoov
GRAVITY = -9.8
URDF_PATH = "/urdf_robot/"
RENDER_WIDTH, RENDER_HEIGHT = 512, 512

class InmoovGymEnv(gym.Env):
    def __init__(self, urdf_path=URDF_PATH, max_steps=1000, srl_model="ground_truth", multi_view=False, seed=0, debug_mode=False):
        self.seed(seed)
        self.urdf_path = urdf_path

        self._observation = None
        self.debug_mode = debug_mode
        self._inmoov = None
        self._observation = None


        self._inmoov_id = -1
        self._tomato_id = -1
        self.max_steps = max_steps
        self._step_counter = 0

        # for more information, please refer to the function _get_tomato_pos
        self._tomato_link_id = 3

        self.srl_model = srl_model
        self.camera_target_pos = (0.0, 0.0, 1.0)
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self.terminated = False
        self.n_contacts = 0
        if debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        # TODO: here, we only use, for the moment, discrete action
        self.action_space = spaces.Discrete
        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self._step_counter = 0
        self.terminated = False
        self.n_contacts = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
        p.setGravity(0, 0, GRAVITY)

        self._inmoov = inmoov.Inmoov(urdf_path=self.urdf_path, positional_control=True)
        self._inmoov_id = self._inmoov.inmoov_id

        self._tomato_id = p.loadURDF(os.path.join(self.urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5])

        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
    def _get_effector_pos(self):
        return self._inmoov.getGroundTruth()

    def _get_tomato_pos(self):
        # 0: the tomato at the bottom
        # 1,2 : the two little tomatoes on top
        # 3: the only lonely tomato on the top
        tomato_pos = p.getLinkState(self._tomato_id, self._tomato_link_id)[0]
        return np.array(tomato_pos)

    def ground_truth(self):
        # a relative position
        return self._get_effector_pos() - self._get_tomato_pos()

    def _reward(self):
        distance = np.linalg.norm(self._get_effector_pos() - self._get_tomato_pos(), 2)
        # printYellow("The distance between target and effector: {:.2f}".format(distance))
        return - distance

    def _termination(self):
        """
        :return: (bool) whether an episode is terminated
        """
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def effector_position(self):
        return self._inmoov.getGroundTruth()

    def step(self, action=None):
        if action is None:
            action = np.array([0, 0, 0])

        tomato_pos = self._get_tomato_pos()
        eff_pos = self._get_effector_pos()
        action = tomato_pos - eff_pos
        self._inmoov.apply_action_pos(action)
        p.stepSimulation()
        self._step_counter += 1
        reward = self._reward()
        obs = self.get_observation()
        done = self._termination()
        infos = {}
        printYellow("reward is : {:.2f}".format(reward))
        return np.array(obs), reward, done, infos
        # printGreen(action)
        # printYellow(self._inmoov.getGroundTruth())
        # tt()
        # self._observation = self.render(mode='rgb')
        # reward = self._reward()
        # done = self._termination()
        # self.obs[:], rewards, self.dones, infos
        # return np.array(self._observation), reward, done, {}

    def get_observation(self):
        if self.srl_model == "raw_pixel":
            self._observation = self.render(mode="rgb")
        elif self.srl_model == "ground_truth":
            self._observation = self.ground_truth()
        else:
            raise NotImplementedError
        return self._observation


    def render(self, mode='rgb'):
        camera_target_position = self.camera_target_pos
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=2.,
            yaw=145,  # 145 degree
            pitch=-36,  # -36 degree
            roll=0,
            upAxisIndex=2
        )
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
        px, depth, mask = np.array(px), np.array(depth), np.array(mask)
        return px

    def close(self):
        return
