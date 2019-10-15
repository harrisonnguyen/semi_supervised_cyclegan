#!/bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/ssl/ --batch-size 1 --n-epochs 50 --cycle-loss-weight 10.0 --summary-freq 500 --learning-rate 2e-4 --mod-a t1 --mod-b flair --model cycle --patch-size 240 --experiment_id 1 #--data-dir /media/hngu4068/8E50BB0C50BAF9D3/brats2018
