# File for policy evaluation

import gym
import tensorflow as tf
import numpy as np
import pybullet as p

from stable_baselines import TD3
from stable_baselines.common.evaluation import evaluate_policy

from rex_gym.envs.rex_gym_env import RexGymEnv
from rexPeriodicRewardEnv import rexPeriodicRewardEnv

env = rexPeriodicRewardEnv(render=True)
p = env.rex._pybullet_client
model_id = env.rex.quadruped
print('model ID: ', model_id)
rex_joints = p.getNumJoints(bodyUniqueId=model_id)
print('Number of Rex Joints = ', rex_joints)

link_name_to_ID = {}
for i in range(rex_joints):
	name = p.getJointInfo(model_id, i)[12].decode('UTF-8')
	link_name_to_ID[name] = i

# Load the model
model = TD3.load(load_path='training_models/TD3_PR_checkpoint_models_3/rl_model_1420000_steps', env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

# Enjoy trained agent (not really that enjoyable TBH)
obs = env.reset()
des = np.array([0,0,0,1])
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    for l in link_name_to_ID.keys():
        if 'shoulder' in l:
            print('Link name = ', l)
            orientation = np.array(p.getLinkState(bodyUniqueId=model_id, linkIndex=link_name_to_ID[l])[-1])
            print('Link orientation (Quat) = ', orientation) 
            similarity = 1 - np.inner(des, orientation)**2
            print('Similarity error = ', similarity)
    #print(info)
    # print('Observations = ', obs)
    #print('Done = ', dones)
    # print('Reward = ', rewards)

    #print('Rex base pos = ', env.rex.GetBasePosition())
    #print('Rex base orientation = ', env.rex.GetBaseOrientation())
    if dones:
        env.reset()

env.close()