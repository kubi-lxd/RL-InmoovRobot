import os
import time

import pybullet as p
import pybullet_data
import numpy as np
from ipdb import set_trace as tt
# from environments.inmoov.inmoov import Inmoov
# from environments.inmoov.inmoov_p2p import InmoovGymEnv
from environments.inmoov.inmoov_p2p_client_ready import InmoovGymEnv
from environments.inmoov.inmoov_view_test import Inmoov


def test_inmoov():
    robot = Inmoov(debug_mode=True)
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # 这是个足球场的环境， 注释掉上面一行，运行下一行可见
    # stadiumId = p.loadSDF(os.path.join(_urdf_path, "stadium.sdf"))
    sjtu_urdf_path = "/urdf_robot/"
    p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5],
                           baseOrientation=[0, 0, 0, 1])
    robot.robot_render()
    while True:  # 根本停不下来！！！！
        time.sleep(0.02)
        robot.debugger_step()
        # 通过位置偏移来控制机器人末端
        # robot.apply_action_pos(motor_commands=[0.0, 0.02, -0.01])
        # 通过关节角度控制机器人末端
        # robot.apply_action_joints(motor_commands=np.random.uniform(-1,1, 53)


def test_inmoov_gym():
    robot = InmoovGymEnv(debug_mode=False, render=False)
    while True:
        time.sleep(0.5)
        # robot.step([-0.1,0.0,0.0])
        robot.server_step([0.7, -0.01, 1.6802])
        # robot.step()

def test_robot_view():
    robot = Inmoov(debug_mode=False, )
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # 这是个足球场的环境， 注释掉上面一行，运行下一行可见
    # stadiumId = p.loadSDF(os.path.join(_urdf_path, "stadium.sdf"))
    sjtu_urdf_path = "/urdf_robot/"
    p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5],
               baseOrientation=[0, 0, 0, 1])
    robot.robot_render()
    while True:  # 根本停不下来！！！！
        time.sleep(0.02)
        robot.robot_render()
        robot.debugger_step()

if __name__ == '__main__':
    test_inmoov_gym()
    # test_inmoov()
    # test_robot_view()