#!/bin/bash

python train.py --algo=ppo --procs=8 --frame=1e7 --log-interval=1000 --save-interval=1000 \
  --tb --lr=1e-4 --recurrence
