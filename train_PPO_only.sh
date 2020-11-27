#!/bin/bash

python entry_point.py ppo  --save_actor rex_actor.pt --save_critic rex_critic.pt \
  --entropy_coeff 0.7 --seed 0 --prenormalize_steps 1000 --kl 0.1 --arch ff
