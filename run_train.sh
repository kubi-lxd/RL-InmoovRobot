#!/bin/bash
python -m rl_baselines.train --algo ppo2 --log-dir logs/lxdtest/ --srl-model ground_truth -c --no-vis --num-cpu 4 --num-timesteps 10000 --env InmoovOneArmButtonGymEnv-v0
