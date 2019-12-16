#!/bin/bash
# python -m rl_baselines.train --algo ppo2 --log-dir logs/lxdtest/ --srl-model ground_truth -c --no-vis --num-cpu 6 --num-timesteps 10000 --env InmoovOneArmButtonGymEnv-v0
python -m rl_baselines.train --env InmoovGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-cpu 6 --num-timesteps 2000000