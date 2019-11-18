import os
import time

import pybullet as p
import pybullet_data
import numpy as np

from environments.inmoov.inmoov import Inmoov

if __name__ == '__main__':
    robot = Inmoov(debug_mode=False)
    # _urdf_path = pybullet_data.getDataPath()
    _urdf_path = "/home/tete/work/SJTU/inmoov/robotics-rl-srl/pybullet_data"
    planeId = p.loadURDF(os.path.join(_urdf_path, "plane.urdf"))
    # stadiumId = p.loadSDF(os.path.join(_urdf_path, "stadium.sdf"))
    sjtu_urdf_path = "/home/tete/work/SJTU/kuka_play/robotics-rl-srl/urdf_robot"
    #tomato1Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0,1,0.5] )
    tomato2Id = p.loadURDF(os.path.join(sjtu_urdf_path, "tomato_plant.urdf"), [0.4, 0.4, 0.5], baseOrientation=[0,0,0,1])
    # robot = Inmoov()
    num_joint = robot.get_action_dimension()
    i = 0
    # while True:
    #     time.sleep(0.01)
    #     robot.debugger_step()
        # print("Step {}".format(i))
        # i+=1
        # robot.apply_action([3,3,3,-10,3])
        # robot.render(2)

    p.disconnect()