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

    robot.get_joint_info()
    time1 = time.time()
    k = 1
    while True:
        i += 1
        # robot.debugger_step()
        # print("Step {}".format(i))
        # i+=1
        robot.apply_action_pos([3,3,3,-10,3])
        # robot.render(2)
        if i%1000 == 0:
            print((time.time() -time1) / k)
            k +=1
    p.disconnect()