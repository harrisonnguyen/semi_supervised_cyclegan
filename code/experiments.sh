#!/bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/ssl/ --batch-size 1 --n-epochs 1 --cycle-loss-weight 10.0 --summary-freq 50 --learning-rate 2e-4 --mod-a t2 --mod-b flair --model semi --patch-size 240 --depth 3 --experiment-id 2 --semi-loss-weight 10.0 #--data-dir /media/hngu4068/8E50BB0C50BAF9D3/brats2018
