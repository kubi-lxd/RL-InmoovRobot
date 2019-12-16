#!/bin/bash
read -p "model file name(ExampleInput:19-12-15_08h00_22):" ModelPATH
python -m replay.enjoy_baselines --log-dir logs/InmoovGymEnv-v0/ground_truth/ppo2/$ModelPATH/ --render --action-proba
