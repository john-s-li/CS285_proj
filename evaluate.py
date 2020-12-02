# File for policy evaluation

import gym
import tensorflow as tf
import numpy as np

from stable_baselines import TD3
from stable_baselines.common.evaluation import evaluate_policy

from rex_gym.envs.rex_gym_env import RexGymEnv

env = RexGymEnv(render=True, signal_type='ol')

# Load the model
model = TD3.load(load_path='Archive/TD3_model_logs/best_model', env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

# Enjoy trained agent (not really that enjoyable TBH)
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(info)
    # print('Observations = ', obs)
    print('Done = ', dones)
    # print('Reward = ', rewards)

    print('Rex base pos = ', env.rex.GetBasePosition())
    print('Rex base orientation = ', env.rex.GetBaseOrientation())

env.close()