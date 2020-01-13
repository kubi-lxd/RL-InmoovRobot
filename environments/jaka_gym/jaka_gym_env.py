import os
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.jaka_gym.jaka import Jaka
from environments.srl_env import SRLGymEnv
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224
URDF_PATH = "/urdf_robot/jaka_urdf/jaka_local.urdf"

def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class JakaGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=1000,
                 env_rank=0,
                 srl_pipe=None, is_discrete=True,
                 action_repeat=1, srl_model="ground_truth",
                 control_mode = "position",
                 seed=0, debug_mode=False, **_):
        super(JakaGymEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=True,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)

        self.seed(seed)
        self.urdf_path = urdf_path

        self._observation = None
        self.debug_mode = debug_mode
        self._inmoov = None
        self._observation = None
        self.action_repeat = action_repeat
        self._jaka_id = -1
        self._button_id= -1
        self.max_steps = max_steps
        self._step_counter = 0
        self._render = False
        self.position_control = (control_mode == "position")
        self.discrete_action = is_discrete
        self.srl_model = srl_model
        self.camera_target_pos = (0.0, 0.0, 1.0)
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self.terminated = False
        self.n_contacts = 0
        self.state_dim = self.getGroundTruthDim()
        self._first_reset_flag = False

    def reset(self, generated_observation=None, state_override=None):
        self._step_counter = 0
        self.terminated = False
        self.n_contacts = 0
        if not self._first_reset_flag:
            p.resetSimulation()
            self._first_reset_flag = True
            p.setPhysicsEngineParameter(numSolverIterations=150)
            p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
            p.setGravity(0, 0, GRAVITY)

            self._jaka = Jaka(urdf_path=self.urdf_path, positional_control=self.position_control)
            self._jaka_id = self._jaka.jaka_id

            self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
            self.button_pos = np.array([x_pos, y_pos, Z_TABLE])

        print('fast reset,resetting robot joints')
        self._jaka.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        return self.get_observation()

    def getSRLState(self, observation):
        return

    def getGroundTruth(self):
        return

    def _reward(self):
        raise NotImplementedError()

    def _termination(self):
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def getGroundTruthDim(self):
        if self.position_control:
            return 3
        else:
            return 6



    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        if self._jaka.positioinal_control and self.discrete_action:
            dv = 1.
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action]
            action = [dx, dy, dz]
            for _ in range(self.action_repeat):
                self._jaka.apply_action_pos(action)
                p.stepSimulation()
            self._step_counter += 1
            reward = self._reward()
            obs = self.get_observation()
            done = self._termination()
            infos = {}
            return np.array(obs), reward, done, infos





if __name__ == '__main__':
    jaka = JakaGymEnv()
    i = 0
    while i < 100:
        i += 1
        jaka.step(1)
        jaka._render(img_name='trash/out{}.png'.format(i))
