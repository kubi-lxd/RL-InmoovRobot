import argparse
import os
import time
from visdom import Visdom

from environments.inmoov.inmoov_p2p import InmoovGymEnv
from gym.envs.registration import registry, patch_deprecated_methods, load


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for reinforcement learning")
    parser.add_argument('--algo', type=str, default='ppo2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-timsteps', type=int, default=int(1e5))
    parser.add_argument('--obs-type', type=str, default='ground_truth')
    args, unknown = parser.parse_known_args()

    env =
