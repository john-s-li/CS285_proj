# Entry Point file utilizing stable baselines from OpenAI

import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback

from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from rex_gym.envs.rex_gym_env import RexGymEnv
from rex_gym.envs.gym.walk_env import RexWalkEnv

from rexPeriodicRewardEnv import rexPeriodicRewardEnv

use_PPO = True
use_TD3 = False

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
if use_PPO:
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
                tensorboard_log='PPO_periodic_reward_logs/',
                verbose=1, # the verbosity level: 0 none, 1 training information, 2 tensorflow debug
                seed=0)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(env, 
                                best_model_save_path='PPO2_periodic_reward_model_logs/',
                                log_path='PPO2_periodic_reward_logs_1/', eval_freq=500,
                                deterministic=True, render=False)

    model.learn(total_timesteps=int(1e9), callback=eval_callback, log_interval=100) # Try a billion learning steps

    model.save("ppo2_rex_1_billion") # save the last model in training

if use_TD3:
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

    eval_callback = EvalCallback(env, 
                            best_model_save_path='TD3_model_logs/',
                            log_path='TD3_logs_1/', eval_freq=2000,
                            deterministic=True, render=False)

    # model = TD3.load(load_path='TD3_best_so_far/best_model', 
    #                 env=env,
    #                 tensorboard_log='TD3_experimental_logs/')
    
    model = TD3(CustomTD3Policy, 
                env, 
                gamma=0.99, 
                learning_rate=3e-4, 
                buffer_size=100000,
                learning_starts=100000,
                train_freq=100, 
                gradient_steps=100, 
                batch_size=256, 
                tau=0.005, 
                policy_delay=2, 
                target_policy_noise=0.2, 
                target_noise_clip=0.5, 
                random_exploration=0.0, 
                tensorboard_log='TD3_experimental_logs/',
                seed=1, 
                action_noise=action_noise, 
                verbose=1)

    model.learn(total_timesteps=int(1e7), # 10 million training steps
                callback=eval_callback, 
                log_interval=100)

    model.save("td3_rex")
