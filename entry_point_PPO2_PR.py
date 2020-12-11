import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from rex_gym.envs.rex_gym_env import RexGymEnv
from rexPeriodicRewardEnv import rexPeriodicRewardEnv


env = rexPeriodicRewardEnv(render=False)
# env = DummyVecEnv([lambda: env])
# check to see that rex_gym is compatible with baselines
check_env(env) 

# Make parallel environments of rex
n_envs = 1
#env = DummyVecEnv([ lambda:env for i in range(n_envs) ])
#env = VecNormalize(env, 
                # norm_obs=True, 
                # norm_reward=False,
                # clip_obs=5., # this is the value used in ppo/algorithm.py from Nico 
                # gamma=0.99) # with or without VecNormalize

# use PPO2 for use of recurrent policies to combat partial observability

policy_kwargs = dict(layers=[256, 256])

print('Running Rex with PPO2 \n')
model = PPO2(MlpLstmPolicy, # policy
            env, 
            gamma=0.99, # discount factor
            n_steps=128, # number of steps to run for each environment per update
            ent_coef=0.03, # entropy coefficient for exploration
            learning_rate=0.00025, 
            vf_coef=0.5, # value function coefficient for the loss calculation
            max_grad_norm=0.5, # maximum value for the gradient clipping
            lam=0.95, # factor for trade-off of bias vs variance for Generalized Advantage Estimator
            nminibatches=n_envs*1, # number of training minibatches per update. For recurrent policies, the number of environments run in parallel should be a multiple of nminibatches.
            noptepochs=10, # number of epoch when optimizing the surrogate
            cliprange=0.2, # this is epsilon for L_clip in PPO paper
            tensorboard_log='PPO_PR_logs/',
            verbose=1, # the verbosity level: 0 none, 1 training information, 2 tensorflow debug
            seed=0,
            policy_kwargs=policy_kwargs)

# Use deterministic actions for evaluation
# eval_callback = EvalCallback(env, 
#                             best_model_save_path='PPO2_periodic_reward_models_2/',
#                             log_path='PPO2_periodic_reward_logs_2/', eval_freq=1000,
#                             deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='PPO2_PR_checkpoint_models/')
#callback = CallbackList([checkpoint_callback, eval_callback])

model.learn(total_timesteps=int(1e7), callback=checkpoint_callback, log_interval=50000) # Try 1 million

model.save("ppo2_rex_PR") # save the last model in training