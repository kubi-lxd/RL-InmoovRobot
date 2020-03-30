import zmq, sys
from zmq import ssh
from matplotlib import pyplot as plt
import numpy as np
import time
from ipdb import set_trace as tt

from .joints_registry import joint_info
# from environments.inmoov.inmoov_p2p_client_ready import InmoovGymEnv

SERVER_PORT = 7777
HOSTNAME = 'localhost'
SSH_NAME = "******" # SSH ip
SSH_PORT = 00000 # SSH port
SSH_PWD = "*****" # SSH passwords
USER_NAME = "tete" # SSH username

def server_connection():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:{}".format(SERVER_PORT))
    print("Waiting for the client")
    line = "Hello Big Man"
    socket.send_json({'msg': line})
    msg = socket.recv_json()
    assert 'msg' in msg and msg['msg'] == line, "Connection failed, check server and client configuration"
    print("Client Connected")
    return socket

def client_connection():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))
    print("Waiting for server")
    msg = socket.recv_json()
    # resend message to ensure the integrity of the msg
    socket.send_json(msg)
    print("Server Connected")
    return socket

def client_ssh_connection():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    password = SSH_PWD
    username = USER_NAME
    server_port = SSH_PORT
    server_name = SSH_NAME
    server = "{}@{}:{}".format(username, server_name, server_port)
    addr = "tcp://{}:{}".format(HOSTNAME, SERVER_PORT)
    tunnuel = ssh.tunnel_connection(socket, addr=addr, server=server, password=password)
    print("Waiting for server")
    msg = socket.recv_json()
    # resend message to ensure the integrity of the msg
    socket.send_json(msg)
    print("send: {}".format(msg))
    print("Server Connected")
    return socket

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def plot_robot_view(left_px, right_px):
    plt.ion()
    figsize = np.array([2, 1]) * 3
    fig = plt.figure("Inmoov", figsize=figsize)
    ax1 = fig.add_subplot(121)
    ax1.imshow(left_px)
    ax1.set_title("Left")
    ax2 = fig.add_subplot(122)
    ax2.imshow(right_px)
    ax2.set_title("Right")
    fig.suptitle('Inmoov Simulation: Two Cameras View', fontsize=20)
    plt.draw()
    plt.pause(0.00001)

if __name__ == "__main__":
    socket = client_ssh_connection()
    # context = zmq.Context()
    # socket = context.socket(zmq.PAIR)
    # # socket.bind("tcp://*:{}".format(SERVER_PORT))
    # print("Connecting")
    # # HOSTNAME = "283a60820s.wicp.vip"
    # # SERVER_PORT = 17253
    # socket.bind("tcp://*:{}".format(SERVER_PORT))
    # # ssh.tunnel_connection(socket, "")
    # print("Connected")

    # get joint information
    joint_info = np.array([p[:1]+p[2:] for p in joint_info])
    # joint_low_limit, joint_high_limit = np.zeros(shape=(53,)), np.ones(shape=(53,))
    joint_low_limit, joint_high_limit = joint_info[:, 1], joint_info[:, 2]
    step = 0
    # TODO: please modify this part to control the robot
    while True:
        time.sleep(0.2)
        print("Step: {}".format(step))
        step += 1
        position = np.random.uniform(low=joint_low_limit*0.1, high=joint_high_limit*0.1)
        msg = {"command":"position", "position": position.tolist()}
        socket.send_json(msg)
        step_data = []
        for i in range(5):
            step_data.append(recv_array(socket))
        joint_state = step_data[0]
        left_px, right_px = step_data[3][0], step_data[3][1]
        reward = np.squeeze(step_data[1])
        done = np.squeeze(step_data[2])
        effector_position = step_data[4]
        # print(joint_state - position)
        plot_robot_view(left_px, right_px)

