#!/bin/bash

python3 entry_point.py ppo  --save_actor rex_actor.pt --save_critic rex_critic.pt \
  --entropy_coeff 0.3 --seed 1 --prenormalize_steps 1000
