import argparse
import os
import time
from visdom import Visdom
import inspect

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from environments.inmoov.inmoov_p2p import InmoovGymEnv
from gym.envs.registration import registry, patch_deprecated_methods, load

from ipdb import set_trace as tt

from stable_baselines.common.vec_env import VecEnv, VecNormalize, DummyVecEnv, SubprocVecEnv, VecFrameStack
from environments.utils import makeEnv

def createEnvs(args, allow_early_resets=False, env_kwargs=None, load_path_normalise=None):
    """
    :param args: (argparse.Namespace Object)
    :param allow_early_resets: (bool) Allow reset before the enviroment is done, usually used in ES to halt the envs
    :param env_kwargs: (dict) The extra arguments for the environment
    :param load_path_normalise: (str) the path to loading the rolling average, None if not available or wanted.
    :return: (Gym VecEnv)
    """
    # imported here to prevent cyclic imports

    envs = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=allow_early_resets, env_kwargs=env_kwargs)
            for i in range(args.num_cpu)]

    if len(envs) == 1:
        # No need for subprocesses when having only one env
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    envs = VecFrameStack(envs, args.num_stack)

    envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    # envs = loadRunningAverage(envs, load_path_normalise=load_path_normalise)

    return envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for reinforcement learning")
    parser.add_argument('--algo', type=str, default='ppo2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='/logs/')
    parser.add_argument('--env', type=str, default='KukaButtonGymEnv-v0')
    parser.add_argument('--num-timsteps', type=int, default=int(1e5))
    parser.add_argument('--obs-type', type=str, default='ground_truth')
    parser.add_argument('--num-cpu', type=int, default=1)
    args, unknown = parser.parse_known_args()

    env_class = InmoovGymEnv
    # env default kwargs
    default_env_kwargs = {k: v.default
                          for k, v in inspect.signature(env_class.__init__).parameters.items()
                          if v is not None}
    env = createEnvs(args)
    tt()
    model = PPO2(policy=MlpPolicy, env=env, learning_rate=lambda f: f*2.5e-4, verbose=1)
    model.learn(total_timesteps=args.num_timsteps, seed=args.seed)
