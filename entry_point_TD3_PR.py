# Entry Point file utilizing stable baselines from OpenAI

import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rexPeriodicRewardEnv import rexPeriodicRewardEnv


env = rexPeriodicRewardEnv(terrain_id='plane', render=False) 
# env = DummyVecEnv([lambda: env])
# check to see that rex_gym is compatible with baselines
#check_env(env) 

# Make parallel environments of rex
n_envs = 1
#env = DummyVecEnv([ lambda:env for i in range(n_envs) ])
#env = VecNormalize(env, 
                # norm_obs=True, 
                # norm_reward=False,
                # clip_obs=5., # this is the value used in ppo/algorithm.py from Nico 
                # gamma=0.99) # with or without VecNormalize

# use PPO2 for use of recurrent policies to combat partial observability


from stable_baselines.td3.policies import FeedForwardPolicy

class CustomTD3Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                                layers=[256, 256],
                                                layer_norm=True,
                                                feature_extraction="mlp")
        
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# eval_callback = EvalCallback(env, 
#                         best_model_save_path='TD3_PR_best_models_2/',
#                         log_path='TD3_eval_PR_logs_2/', eval_freq=50000,
#                         deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='TD3_PR_checkpoint_models_4/')
# callback = CallbackList([checkpoint_callback, eval_callback])

model = TD3(CustomTD3Policy, 
            env, 
            gamma=0.99, 
            learning_rate=0.0001, 
            buffer_size=100000,
            learning_starts=10000,
            train_freq=100, 
            gradient_steps=100, 
            batch_size=256, 
            tau=0.005, 
            policy_delay=2, 
            target_policy_noise=0.2, 
            target_noise_clip=0.5, 
            random_exploration=0.1, 
            tensorboard_log='TD3_experiment_PR_logs_4/',
            seed=1, 
            action_noise=action_noise, 
            verbose=1)

model.learn(total_timesteps=int(1e7), # 10 mil training steps
            callback=checkpoint_callback, 
            log_interval=5000)

model.save("td3_rex_PR")
