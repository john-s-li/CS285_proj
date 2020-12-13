import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG

from rex_gym.envs.rexPeriodicRewardEnv import rexPeriodicRewardEnv

def eval_policy(policy, env, eval_episodes=10):
	
	rewards = []
	for _ in range(eval_episodes):
		state, done = env.reset(), False
		env.seed(np.random.randint(5, 500))
		i = 0
		sum_reward = 0
		while (not done and i < env._max_episode_steps):
			action = policy.select_action(np.array(state))
			state, reward, done, _ = env.step(action)
			sum_reward += reward
			i += 1
		rewards.append(sum_reward)

	mean = np.mean(rewards)
	std = np.std(rewards)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: eval_reward = {mean:.3f} and eval_std = {std:.3f}")
	print("---------------------------------------")
	return (mean, std)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default=None)                      # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seed
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print("Evaluation of TD3 policy")
	print("---------------------------------------")

	env = rexPeriodicRewardEnv(render=True)
	
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	
	if args.load_model != "":
		policy_file = f"./models/{file_name}" if args.load_model == "default" else args.load_model
		policy.load(policy_file)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate the policy
	eval_policy(policy, env)

