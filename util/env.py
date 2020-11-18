import os
import time 
import torch
import numpy as np

import gym
from rex_gym.envs.rex_gym_env import RexGymEnv

def env_factory(path, verbose=False, **kwargs):
    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.

    Modified to work with rex_gym environment

    """
    from functools import partial
    from rex_gym.envs.rex_gym_env import RexGymEnv
    
    return partial(RexGymEnv,terrain_id='plane', render=False)

def eval_policy(policy, min_timesteps=1000, max_traj_len=1000, visualize=True, env=None, verbose=True):
  env_name = env
  with torch.no_grad():
    if env_name is None:
      env = env_factory(policy.env_name)()
    else:
      env = env_factory(env_name)()

    print("Policy is a: {}".format(policy.__class__.__name__))
    reward_sum = 0
    env.dynamics_randomization = False
    total_t = 0
    episodes = 0

    obs_states = {}
    mem_states = {}

    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0
      eval_reward = 0
      episodes += 1

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        if (hasattr(env, 'simrate') or hasattr(env, 'dt')) and visualize:
          start = time.time()

        env.speed = 0.0
        env.side_speed = 0.0
        env.phase_add = 60
        env.orient_add = 0
        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        if visualize:
          env.render()
        eval_reward += reward
        timesteps += 1
        total_t += 1

        if hasattr(policy, 'get_quantized_states'):
          obs, mem = policy.get_quantized_states()
          obs_states[obs] = True
          mem_states[mem] = True
          print(policy.get_quantized_states(), len(obs_states), len(mem_states))

        if visualize:
          if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

          if hasattr(env, 'dt'):
            while time.time() - start < env.dt:
              time.sleep(0.0005)

      reward_sum += eval_reward
      if verbose:
        print("Eval reward: ", eval_reward)
    return reward_sum / episodes

def train_normalizer(policy, min_timesteps, max_traj_len=1000, noise=0.5):
  with torch.no_grad():
    env = env_factory(policy.env_name)()
    env.dynamics_randomization = False

    total_t = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        if noise is None:
          action = policy.forward(state, update_norm=True, deterministic=False).numpy()
        else:
          action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1
