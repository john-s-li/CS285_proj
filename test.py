# Test file for importing quadruped into PyBullet

import gym
from rex_gym.envs.rex_gym_env import RexGymEnv

env = RexGymEnv(terrain_id='plane') # not sure why this has to be specified

for _ in range(1000):
    ob, re, done, ac = env.step(env.action_space.sample()) # take a random action
env.close()