import math
import os

import numpy as np
import pybullet as p
import pybullet_data
from ipdb import set_trace as tt

# URDF_PATH = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot/"

URDF_PATH="/home/kunan/SRL/newp/robotics-rl-srl/urdf_robot/"
GRAVITY = -10.

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

import time
if __name__ == '__main__':
    p.connect(p.GUI)
    robot = Inmoov()
    _urdf_path = pybullet_data.getDataPath()
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # robot = Inmoov()
    for i in range(1000000):
        robot.debugger_step()

    p.disconnect()
