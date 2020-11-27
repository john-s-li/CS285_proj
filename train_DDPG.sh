#!/bin/bash

python entry_point.py ddpg  --save_actor ddpg_actor.pt --save_critic ddpg_critic.pt \
  --arch ff 
