import math
import os

import numpy as np
import pybullet as p
import pybullet_data
from ipdb import set_trace as tt

# URDF_PATH = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot/"
URDF_PATH="../../urdf_robot/"
GRAVITY = -10.

'''
class Inmoov:
    def __init__(self, urdf_path=URDF_PATH):
        self.urdf_path = urdf_path
        self._renders = True
        self.debug_mode = True

        self.reset()

        if self.debug_mode:
            debug_joints = []
            for j in range(self.num_joints):
                debug_joints.append(p.addUserDebugParameter("joint_{}".format(j+60), -1., 1., 0))
            self.debug_joints = debug_joints


    def reset(self):
        """
        Reset the environment
        """
        p.setGravity(0., 0., GRAVITY)
        self.inmoov_id = p.loadURDF(os.path.join(self.urdf_path, 'inmoov_col.urdf'))
        self.num_joints = p.getNumJoints(self.inmoov_id)
        # tmp1 = p.getNumBodies(self.inmoov_id)  # Equal to 1, only one body
        # tmp2 = p.getNumConstraints(self.inmoov_id)  # Equal to 0, no constraint?
        # tmp3 = p.getBodyUniqueId(self.inmoov_id)  # res = 0, do not understand
        for jointIndex in range(self.num_joints):
            p.resetJointState(self.inmoov_id, jointIndex, 0.1 )

        # tt( )

    def debugger_step(self):
        useful_joints = [2, 4, 7]
        if self.debug_mode:
            current_joints = []
            for j in self.debug_joints:
                tmp_joint_control = p.readUserDebugParameter(j)
                current_joints.append(tmp_joint_control)
            for jointIndex, joint_state in enumerate(current_joints):
                try:
                    p.resetJointState(self.inmoov_id, jointIndex+60, targetValue=joint_state)
                except:
                    continue
            p.stepSimulation()
'''

from environments.inmoov.inmoov import Inmoov

import time
if __name__ == '__main__':
    robot = Inmoov(debug_mode=True)
    # _urdf_path = pybullet_data.getDataPath()
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # stadiumId = p.loadSDF(os.path.join(_urdf_path, "stadium.sdf"))
    sjtu_urdf_path = "../../urdf_robot/"
    # tomato1Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0,1,0.5] )
    tomato2Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5],
                           baseOrientation=[0, 0, 0, 1])
    # robot = Inmoov()
    num_joint = robot.get_action_dimension()
    i = 0

    robot.get_joint_info()
    time1 = time.time()
    k = 1
    while True:
        time.sleep(0.02)
        robot.debugger_step()
        print("Step {}".format(i))
        i += 1
        robot.apply_action_pos(motor_commands=[-0.1, 0., -0.05])
        robot.render(2)
        if i % 100 == 0:
            robot.reset()
            print((time.time() - time1) / k)
            k += 1
    p.disconnect()
