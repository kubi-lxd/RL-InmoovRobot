import os

import numpy as np
from matplotlib import pyplot as plt
import pybullet as p
from ipdb import set_trace as tt

from util.color_print import printGreen, printBlue, printRed, printYellow
from environments.inmoov.joints_registry import joint_registry, control_joint
URDF_PATH = "/urdf_robot/"
GRAVITY = -9.8
RENDER_WIDTH, RENDER_HEIGHT = 512, 512
CONTROL_JOINT = list(control_joint.keys())
CONTROL_JOINT.sort()


class Inmoov:
    def __init__(self, urdf_path=URDF_PATH, positional_control=True, debug_mode=False, use_null_space=True):
        self.urdf_path = urdf_path
        self._renders = True
        self.debug_mode = debug_mode
        self.inmoov_id = -1
        self.num_joints = -1
        self.robot_base_pos = [0, 0, 0]
        # effectorID = 28: right hand
        # effectorID = 59: left hand
        self.effectorId = 28
        self.effector_pos = None
        # joint information
        # jointName, (jointLowerLimit, jointUpperLimit), jointMaxForce, jointMaxVelocity, linkName, parentIndex
        self.joint_name = {}
        self.joint_lower_limits = None
        self.joint_upper_limits = None
        self.jointMaxForce = None
        self.jointMaxVelocity = None
        self.joints_key = None
        # camera position
        self.camera_target_pos = (0.0, 0.0, 1.0)
        # Control mode: by joint or by effector position
        self.positional_control = positional_control
        # inverse Kinematic solver, ref: Pybullet
        self.use_null_space = use_null_space
        if self.debug_mode:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(5., 180, -41, [0.52, -0.2, -0.33])

            # To debug the joints of the Inmoov robot
            debug_joints = []
            self.joints_key = []
            for joint_index in joint_registry:
                self.joints_key.append(joint_index)
                debug_joints.append(p.addUserDebugParameter(joint_registry[joint_index], -1., 1., 0))
            self.debug_joints = debug_joints
        else:
            p.connect(p.DIRECT)
        global CONNECTED_TO_SIMULATOR
        CONNECTED_TO_SIMULATOR = True
        self.reset()

    def reset(self):
        """
        Reset the environment
        """
        # p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=150)
        # p.setGravity(0., 0., GRAVITY)

        self.inmoov_id = p.loadURDF(os.path.join(self.urdf_path, 'inmoov_col.urdf'), self.robot_base_pos)
        self.num_joints = p.getNumJoints(self.inmoov_id)
        self.get_joint_info()

        # tmp1 = p.getNumBodies(self.inmoov_id)  # Equal to 1, only one body
        # tmp2 = p.getNumConstraints(self.inmoov_id)  # Equal to 0, no constraint?
        # tmp3 = p.getBodyUniqueId(self.inmoov_id)  # res = 0, do not understand
        for jointIndex in self.joints_key:
            p.resetJointState(self.inmoov_id, jointIndex, 0.)
        # get the effector world position
        self.effector_pos = p.getLinkState(self.inmoov_id, self.effectorId)[0]
        # # get link information
        # ######################## debug part #######################
        # from mpl_toolkits.mplot3d import Axes3D
        # #To plot the link index by graphical representation
        # link_position = []
        # p.getBasePositionAndOrientation(self.inmoov_id)
        # for i in range(100):
        #     print("linkWorldPosition, , , , workldLinkFramePosition", i)
        #     link_state = p.getLinkState(self.inmoov_id, i)
        #     if link_state is not None:
        #         link_position.append(link_state[0])
        #
        # link_position = np.array(link_position).T
        # print(link_position.shape)
        #
        # fig = plt.figure("3D link plot")
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(link_position[0], link_position[1], link_position[2], c='r', marker='o')
        # for i in range(link_position.shape[1]):
        #     # ax.annotate(str(i), (link_position[0,i], link_position[1,i], link_position[2,i]) )
        #     ax.text(link_position[0,i], link_position[1,i], link_position[2,i], str(i))
        # # ax.set_xlim([-1, 1])
        # # ax.set_ylim([-1, 1])
        # ax.set_xlim([-.25, .25])
        # ax.set_ylim([-.25, .25])
        # ax.set_zlim([1, 2])
        # plt.show()
        # ####################### debug part #######################

    def getGroundTruth(self):
        if self.positional_control:
            position = p.getLinkState(self.inmoov_id, self.effectorId)[0]
            return np.array(position)
        else:  # control by joint and return the joint state (joint position)
            # we can add joint velocity as joint state, but here we didnt, getJointState can get us more infomation
            joints_state = p.getJointStates(self.inmoov_id, self.joints_key)
            return np.array(joints_state)[:, 0]

    def getGroundTruthDim(self):
        if self.positional_control:
            return 3
        else:
            return len(self.joints_key)

    # def __del__(self):
    #     if CONNECTED_TO_SIMULATOR:
    #         p.disconnect()

    def step(self, action):
        raise NotImplementedError

    def _termination(self):
        raise NotImplementedError

    def _reward(self):
        raise NotImplementedError

    def get_action_dimension(self):
        """
        To know how many joint can be controlled
        :return: int
        """
        return len(self.joints_key)

    def get_effector_dimension(self):
        """
        :return: three dimension for x, y and z
        """
        return 3

    def apply_action_joints(self, motor_commands):
        """
        Apply the action to the inmoov robot joint
        If the length of commands is inferior to the length of controlable joint, then we only control the first joints
        :param motor_commands:
        """
        assert len(motor_commands) == len(self.joints_key), "Error, please provide control commands for all joints"
        num_control = len(motor_commands)
        target_velocities = [0] * num_control
        position_gains = [0.3] * num_control
        velocity_gains = [1] * num_control
        p.setJointMotorControlArray(bodyUniqueId=self.inmoov_id,
                                    controlMode=p.POSITION_CONTROL,
                                    jointIndices=self.joints_key,
                                    targetPositions=motor_commands,
                                    targetVelocities=target_velocities,
                                    forces=self.jointMaxForce,
                                    positionGains=position_gains,
                                    velocityGains=velocity_gains
                                    )
        # # Same functionality, but upper lines works better
        # for i in range(num_control):
        #     # p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=CONTROL_JOINT[i],
        #     #                         controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],
        #     #                         targetVelocity=0, force=self.max_force,
        #     #                         maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
        #     p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=CONTROL_JOINT[i],
        #                             controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i],
        #                             targetVelocity=0, force=self.max_force,
        #                             maxVelocity=4., positionGain=0.3, velocityGain=1)
        # p.stepSimulation()

    def apply_action_pos(self, motor_commands):
        """
        Apply the action to the inmoov robot joint
        If the length of commands is inferior to the length of controlable joint, then we only control the first joints
        :param motor_commands: [dx, dy, dz]
        """
        # TODO: Add orientation control information for a better physical representation
        # assert len(motor_commands) == (3+4) x,y,z + Quaternion
        assert len(motor_commands) == 3, "Invalid input commands, please use a 3D: x,y,z information"
        dx, dy, dz = motor_commands
        # We observe that the control has error, and the best way to get the current
        # position is to getLinkState instead of save the initial position and do the incremental thing
        joint_position = p.getLinkState(self.inmoov_id, self.effectorId)
        current_state = joint_position[0]
        # printGreen("Current hand position: {}".format(joint_position[0]))
        # printYellow("Current position by accumulate error")
        # printRed("Error: {}".format(np.linalg.norm(np.array(current_state) - np.array(self.effector_pos))))
        self.effector_pos = current_state
        # TODO: it might be better to constraint the target position in a constraint box?
        target_pos = (current_state[0] + dx, current_state[1] + dy, current_state[2] + dz)
        # Compute the inverse kinematics for every revolute joint
        num_control_joints = len(self.joints_key)
        # position at rest
        rest_poses = [0] * num_control_joints
        # don't know what is its influence
        joint_ranges = [4] * num_control_joints
        # the velocity at the target position
        target_velocity = [0] * num_control_joints
        if self.use_null_space:
            # TODO: Add orientation control
            joint_poses = p.calculateInverseKinematics(self.inmoov_id, self.effectorId, target_pos,
                                                       lowerLimits=self.joint_lower_limits,
                                                       upperLimits=self.joint_upper_limits,
                                                       jointRanges=joint_ranges,
                                                       restPoses=rest_poses
                                                       )
        else:  # use regular KI solution
            joint_poses = p.calculateInverseKinematics(self.inmoov_id, self.effectorId, target_pos)
        # printGreen(joint_poses)
        p.setJointMotorControlArray(self.inmoov_id, self.joints_key,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_poses,
                                    targetVelocities=target_velocity,
                                    #  maxVelocities=self.jointMaxVelocity,
                                    forces=self.jointMaxForce)
        p.stepSimulation()
        # for i, index in enumerate(self.joints_key):
        #     joint_info = self.joint_name[index]
        #     jointMaxForce, jointMaxVelocity = joint_info[2:4]
        #
        #     p.setJointMotorControl2(bodyUniqueId=self.inmoov_id, jointIndex=index, controlMode=p.POSITION_CONTROL,
        #                             targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force,
        #                             maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)



    def get_joint_info(self):
        """
        From this we can see the fact that:
        - no joint damping is set
        - some of the joints are reserved???
        - none of them has joint Friction
        - we have 53 revo
        :return:
        """

        self.joints_key = []
        self.joint_lower_limits, self.joint_upper_limits, self.jointMaxForce, self.jointMaxVelocity = [], [], [], []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.inmoov_id, i)
            # if info[7] != 0:
            #     print(info[1], "has friction")
            # if info[6] != 0:
            #     print(info[1], "has damping")
            if info[2] == p.JOINT_REVOLUTE:
                # jointName, (jointLowerLimit, jointUpperLimit), jointMaxForce, jointMaxVelocity, linkName, parentIndex
                # (info[1], (info[8], info[9]), info[10], info[11], info[12], info[16])
                self.joint_name[i] = info[1]
                if info[1] == b'right_bicep':
                    printYellow(info)
                if info[1] == b'left_bicep':
                    printGreen(info)
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])
                self.jointMaxForce.append(info[10])
                self.jointMaxVelocity.append(info[11])
                self.joints_key.append(i)

    def debugger_step(self):
        if self.debug_mode:
            current_joints = []
            # The order is as the same as the self.joint_key
            for j in self.debug_joints:
                tmp_joint_control = p.readUserDebugParameter(j)
                current_joints.append(tmp_joint_control)
            for joint_state, joint_key in zip(current_joints, self.joints_key):
                p.resetJointState(self.inmoov_id, joint_key, targetValue=joint_state)
            p.stepSimulation()

    def debugger_camera(self):
        if self.debug_mode:
            tete = "Stupid"

    def robot_render(self):
        """
        The image from the robot eye
        :return:
        """
        # TODO

    def render(self, num_camera=1):
        if self._renders:
            plt.ion()
            if num_camera == 1:
                figsize = np.array([3, 1]) * 5
            else:
                figsize = np.array([3, 2]) * 5
            fig = plt.figure("Inmoov", figsize=figsize)

            camera_target_position = self.camera_target_pos

            # view_matrix2 = p.computeViewMatrixFromYawPitchRoll(
            #     cameraTargetPosition=(0.316, 0.316, 1.0),
            #     distance=1.2,
            #     yaw=90,  # 145 degree
            #     pitch=-13,  # -36 degree
            #     roll=0,
            #     upAxisIndex=2
            # )
            # proj_matrix2 = p.computeProjectionMatrixFOV(
            #     fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            #     nearVal=0.1, farVal=100.0)
            # (_, _, px2, depth2, mask2) = p.getCameraImage(
            #     width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix2,
            #     projectionMatrix=proj_matrix2, renderer=p.ER_TINY_RENDERER)

            # first camera
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
            # depth: the depth camera, mask: mask on different body ID
            (_, _, px, depth, mask) = p.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
                projectionMatrix=proj_matrix1, renderer=p.ER_TINY_RENDERER)
            px, depth, mask = np.array(px), np.array(depth), np.array(mask)
            # if there are more than one camera, (only two are allowed actually)
            if num_camera == 2:
                view_matrix2 = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=(0.316, 0.316, 1.0),
                    distance=1.2,
                    yaw=90,  # 145 degree
                    pitch=-13,  # -36 degree
                    roll=0,
                    upAxisIndex=2
                )
                proj_matrix2 = p.computeProjectionMatrixFOV(
                    fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                    nearVal=0.1, farVal=100.0)
                (_, _, px2, depth2, mask2) = p.getCameraImage(
                    width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix2,
                    projectionMatrix=proj_matrix2, renderer=p.ER_TINY_RENDERER)
                ax1 = fig.add_subplot(231)
                ax1.imshow(px)
                ax1.set_title("rgb_1")
                ax2 = fig.add_subplot(232)
                ax2.imshow(depth)
                ax2.set_title("depth_1")
                ax3 = fig.add_subplot(233)
                ax3.imshow(mask)
                ax3.set_title("mask_1")
                ax1 = fig.add_subplot(234)
                ax1.imshow(px2)
                ax1.set_title("rgb_2")
                ax2 = fig.add_subplot(235)
                ax2.imshow(depth2)
                ax2.set_title("depth_2")
                ax3 = fig.add_subplot(236)
                ax3.imshow(mask2)
                ax3.set_title("mask_2")
            else:  # only one camera
                ax1 = fig.add_subplot(131)
                ax1.imshow(px)
                ax1.set_title("rgb_1")
                ax2 = fig.add_subplot(132)
                ax2.imshow(depth)
                ax2.set_title("depth_1")
                ax3 = fig.add_subplot(133)
                ax3.imshow(mask)
                ax3.set_title("mask_1")
            # rgb_array = np.array(px)
            # self.image_plot = plt.imshow(rgb_array)
            # self.image_plot.axes.grid(False)
            # plt.title("Inmoov Robot Simulation")
            fig.suptitle('Inmoov Simulation: Two Cameras View', fontsize=32)
            plt.draw()
            # To avoid too fast drawing conflict
            plt.pause(0.00001)
