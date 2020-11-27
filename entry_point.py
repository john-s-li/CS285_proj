import os

import torch
import hashlib
from collections import OrderedDict
from rex_gym.envs.rex_gym_env import RexGymEnv

from util.env import env_factory, eval_policy
from util.logo import print_logo

if __name__ == "__main__":
	import sys, argparse, time, os
	parser = argparse.ArgumentParser()
	print_logo(subtitle="Recurrent Reinforcement Learning for Robotics.")

	if len(sys.argv) < 2:
		print("Usage: python apex.py [option]", sys.argv)
		exit(1)

	if sys.argv[1] == 'eval':
		sys.argv.remove(sys.argv[1])

		parser.add_argument("--policy", default="./trained_models/ddpg/ddpg_actor.pt", type=str)
		parser.add_argument("--env",      default=None, type=str)
		parser.add_argument("--traj_len", default=1000, type=int)
		args = parser.parse_args()

		policy = torch.load(args.policy)

		eval_policy(policy, min_timesteps=100000, env=args.env, max_traj_len=args.traj_len)
		exit()

	if sys.argv[1] == 'extract': # not exactly sure what this is...
		sys.argv.remove(sys.argv[1])
		from algos.extract_dynamics import run_experiment

		parser.add_argument("--policy", "-p", default=None,           type=str)
		parser.add_argument("--workers",      default=4,              type=int)
		parser.add_argument("--points",       default=5000,           type=int)
		parser.add_argument("--iterations",   default=1000,           type=int)
		parser.add_argument("--batch_size",   default=16,             type=int)
		parser.add_argument("--layers",       default="256,256",      type=str) 
		parser.add_argument("--lr",           default=1e-5,           type=float)
		parser.add_argument("--epochs",       default=5,              type=int)
		parser.add_argument("--logdir",       default='logs/extract', type=str)
		parser.add_argument("--redis",                default=None)
		args = parser.parse_args()
		run_experiment(args)
		exit()

	# Utility for running QBN insertion.
	if sys.argv[1] == 'qbn':
		sys.argv.remove(sys.argv[1])
		from algos.qbn import run_experiment

		parser.add_argument("--nolog",        action='store_true')             # store log data or not.
		parser.add_argument("--seed",   "-s", default=0,            type=int)    # random seed for reproducibility
		parser.add_argument("--policy", "-p", default=None,         type=str)
		parser.add_argument("--data",         default=None,         type=str)
		parser.add_argument("--workers",      default=4,            type=int)
		parser.add_argument("--logdir",       default='logs/qbn',   type=str)
		parser.add_argument("--traj_len",     default=1000,         type=int)
		parser.add_argument("--layers",       default="512,256,64", type=str) 
		parser.add_argument("--lr",           default=1e-5,         type=float)
		parser.add_argument("--dataset",      default=100000,       type=int)
		parser.add_argument("--epochs",       default=500,          type=int)      # number of updates per iter
		parser.add_argument("--batch_size",   default=64,           type=int)
		parser.add_argument("--iterations",   default=500,          type=int)
		parser.add_argument("--episodes",     default=64,           type=int)
		args = parser.parse_args()
		run_experiment(args)
		exit()

	# Options common to all RL algorithms.
	# rex_env = RexGymEnv(terrain_id='plane',render=False)
	parser.add_argument("--nolog",                  action='store_true')              # store log data or not.
	parser.add_argument("--arch",           "-r",   default='ff')                     # either ff, lstm, or gru
	parser.add_argument("--seed",           "-s",   default=0,           type=int)    # random seed for reproducibility
	parser.add_argument("--traj_len",       "-tl",  default=1000,        type=int)    # max trajectory length for environment
	parser.add_argument("--env",            "-e",   default='rex_gym',   type=str)    # environment to train on
	parser.add_argument("--layers",                 default="256,256",   type=str)    # hidden layer sizes in policy
	parser.add_argument("--timesteps",      "-t",   default=1e6,         type=float)  # timesteps to run experiment for

	if sys.argv[1] == 'ddpg':
		sys.argv.remove(sys.argv[1])
		"""
		Utility for running Recurrent/Deep Deterministic Policy Gradients.
		"""
		from algos.off_policy import run_experiment
		parser.add_argument("--start_timesteps",        default=1e4,   type=int)      # number of timesteps to generate random actions for
		parser.add_argument("--load_actor",             default=None,  type=str)      # load an actor from a .pt file
		parser.add_argument("--load_critic",            default=None,  type=str)      # load a critic from a .pt file
		parser.add_argument('--discount',               default=0.99,  type=float)    # the discount factor
		parser.add_argument('--expl_noise',             default=0.2,   type=float)    # random noise used for exploration
		parser.add_argument('--tau',                    default=0.01, type=float)     # update factor for target networks
		parser.add_argument("--a_lr",           "-alr", default=1e-5,  type=float)    # adam learning rate for critic
		parser.add_argument("--c_lr",           "-clr", default=1e-4,  type=float)    # adam learning rate for actor
		parser.add_argument("--normalize"       '-n',   action='store_true')          # normalize states online
		parser.add_argument("--batch_size",             default=64,    type=int)      # batch size for policy update
		parser.add_argument("--updates",                default=1,    type=int)       # (if recurrent) number of times to update policy per episode
		parser.add_argument("--eval_every",             default=100,   type=int)      # how often to evaluate the trained policy
		parser.add_argument("--save_actor",             default=None, type=str)
		parser.add_argument("--save_critic",            default=None, type=str)
		parser.add_argument("--prenormalize_steps",     default=10000,         type=int)      

		parser.add_argument("--logdir",                 default="./logs/ddpg/", type=str)

		args = parser.parse_args()
		args.algo = 'ddpg'

		run_experiment(args)

	if sys.argv[1] == 'ppo':
		sys.argv.remove(sys.argv[1])
		"""
		Utility for running Proximal Policy Optimization.

		"""
		from algos.ppo import run_experiment
		parser.add_argument("--prenormalize_steps",     default=10000,         type=int)      
		parser.add_argument("--num_steps",              default=5000,          type=int)      

		parser.add_argument('--discount',               default=0.99,          type=float)    # the discount factor
		parser.add_argument('--std',                    default=0.13,          type=float)    # the fixed exploration std
		parser.add_argument("--a_lr",           "-alr", default=1e-4,          type=float)    # adam learning rate for actor
		parser.add_argument("--c_lr",           "-clr", default=1e-4,          type=float)    # adam learning rate for critic
		parser.add_argument("--eps",            "-ep",  default=1e-6,          type=float)    # adam eps
		parser.add_argument("--kl",                     default=0.02,          type=float)    # kl abort threshold
		parser.add_argument("--entropy_coeff",          default=0.0,           type=float)
		parser.add_argument("--grad_clip",              default=0.05,          type=float)
		parser.add_argument("--batch_size",             default=64,            type=int)      # batch size for policy update
		parser.add_argument("--epochs",                 default=3,             type=int)      # number of updates per iter
		parser.add_argument("--mirror",                 default=0,             type=float)
		parser.add_argument("--sparsity",               default=0,             type=float)

		parser.add_argument("--save_actor",             default=None,          type=str)
		parser.add_argument("--save_critic",            default=None,          type=str)
		parser.add_argument("--workers",                default=4,             type=int)
		parser.add_argument("--redis",                  default=None,          type=str)

		parser.add_argument("--logdir",                 default="./logs/ppo/", type=str)
		args = parser.parse_args()

		run_experiment(args)

	else:
		print("Invalid option '{}'".format(sys.argv[1]))
