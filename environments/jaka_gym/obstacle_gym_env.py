import os
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from util.color_print import printGreen, printBlue, printRed, printYellow
import gym
from environments.jaka_gym.jaka import Jaka
from environments.srl_env import SRLGymEnv
import cv2
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 224, 224
URDF_PATH = "/urdf_robot/jaka_urdf/jaka_local.urdf"

def getGlobals():
    """
    :return: (dict)
    """
    return globals()
# ssh -N -f -L localhost:6006:localhost:8097  tete@283a60820s.wicp.vip -p 17253
# python -m rl_baselines.train --env JakaButtonGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-cpu 16

class JakaButtonObsGymEnv(SRLGymEnv):
    def __init__(self, urdf_path=URDF_PATH, max_steps=250,
                 env_rank=0, random_target=False,
                 srl_pipe=None, is_discrete=True,
                 action_repeat=1, srl_model="ground_truth",
                 control_mode = "position",
                 seed=0, debug_mode=False, **_):
        super(JakaButtonObsGymEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=True,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)

        self.seed(seed)
        self.urdf_path = urdf_path
        self._random_target = random_target
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

        if debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2., 180, -41, [0., -0.1, 0.1])
        else:
            p.connect(p.DIRECT)
        self.action_space = spaces.Discrete(6)
        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:  # Todo: the only possible observation for srl_model is ground truth
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.reset()

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
            x_pos = -0.1
            y_pos = 0.55
            
            if self._random_target:
                x_pos += 0.15 * self.np_random.uniform(-1, 1)
                y_pos += 0.3 * self.np_random.uniform(-1, 1)
            self.obs_x, self.obs_y = 0.3, 0.4
            self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0])
            self.obstacle_id = p.loadURDF("/urdf/long_cylinder.urdf", [self.obs_x, self.obs_y, 0])
            self.button_pos = np.array([x_pos, y_pos, 0])

        self._jaka.reset_joints()
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)
        self._observation = self.get_observation()
        # if self.srl_model == "raw_pixels":
        #     self._observation = self._observation[0]
        return self._observation

    def getTargetPos(self):
        return self.button_pos

    def getGroundTruth(self):
        return self._jaka.getGroundTruth()

    def _reward(self):
        # r = - np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
        contact_button = 10 * int(len(p.getContactPoints(self._jaka_id, self.button_uid)) > 0)
        contact_obs = len(p.getContactPoints(self._jaka_id, self.obstacle_id))

        r = 0
        if contact_obs > 0:
            self.terminated = True
            r = - 10 * contact_obs
        else:
            distance = np.linalg.norm(self.getGroundTruth() - self.getTargetPos())
            if distance < 0.1:
                r += 10
                self.terminated = True
            else:
                r = 1 / distance - (self._step_counter / self.max_steps) * 2
        return r

    def _termination(self):
        # print(s)
        # print(self._step_counter, self._step_counter > self.max_steps)
        if self.terminated or self._step_counter > self.max_steps:
            return True
        return False

    def get_observation(self):
        if self.srl_model == "ground_truth":
            if self.relative_pos:
                return self.getGroundTruth() - self.getTargetPos()
            return self.getGroundTruth()
        elif self.srl_model == "raw_pixels":
            return self.render()[0]
        else:
            return NotImplementedError()

    @staticmethod
    def getGroundTruthDim():
        return 3

    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        sinus_moving = False
        random_moving = True
        assert int(sinus_moving) + int(random_moving) < 2
        if sinus_moving:
            # p.loadURDF()
            maxx, maxy = self.obs_x, self.obs_y
            time_ratio = np.pi*2*(self._step_counter / self.max_steps)
            obs_x, obs_y = np.cos(time_ratio), maxy
            p.resetBasePositionAndOrientation(self.obstacle_id, [obs_x, obs_y, 0], [0,0,0,1])
        if random_moving:
            xnow, ynow = self.obs_x, self.obs_y
            fields_x = [0.3-0.2, 0.3+0.2]
            fields_y = [0.4-0.3, 0.4+0.2]
            obs_step = 0.1
            move_x = np.random.uniform(-0.1,0.1)
            move_y = np.sqrt(obs_step**2-move_x**2) * (2*(np.random.uniform()>0.5)-1)
            self.obs_x = np.clip(xnow + move_x, fields_x[0], fields_x[1])
            self.obs_y = np.clip(ynow + move_y, fields_y[0], fields_y[1])
            p.resetBasePositionAndOrientation(self.obstacle_id, [self.obs_x, self.obs_y, 0], [0, 0, 0, 1])



        if self.position_control and self.discrete_action:
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

    def render(self, mode='channel_last', img_name=None):
        """
        You could change the Yaw Pitch Roll distance to change the view of the robot
        :param mode:
        :return:
        """
        camera_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._jaka.robot_base_pos,
            distance=2.,
            yaw=145,  # 145 degree
            pitch=-40,  # -45 degree
            roll=0,
            upAxisIndex=2
        )
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.15, farVal=100.0)
        # depth: the depth camera, mask: mask on different body ID
        (_, _, px, depth, mask) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=camera_matrix,
            projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)

        px, depth, mask = np.array(px), np.array(depth), np.array(mask)

        px = px.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        depth = depth.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        mask = mask.reshape(RENDER_HEIGHT, RENDER_WIDTH, -1)
        px = px[..., :-1]
        if img_name is not None:
            cv2.imwrite(img_name, px[..., [2, 1, 0]])
        if mode != 'channel_last':
            # channel first
            px = np.transpose(px, [2, 0, 1])
        return px, depth, mask

if __name__ == '__main__':
    jaka = JakaButtonObsGymEnv(debug_mode=True)
    i = 0
    for i in range(10):
        jaka.step(0)
        jaka.step(3)
        jaka.step(3)
        printYellow(jaka._reward())
        # jaka.render(img_name='trash/out{}.png'.format(i))
    import time
    while True:
        i += 1
        # jaka.step(0)
        # jaka.step(3)
        jaka.step(np.random.randint(0,6))
        time.sleep(0.1)
        printYellow(jaka._reward())
        #
        jaka.render(img_name='trash/out{}.png'.format(i))
