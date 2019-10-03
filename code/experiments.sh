#! /bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/ssl_brats/cycle --batch-size 1 --n-epochs 50 --cycle-loss-weight 5.0 --summary-freq 100 --learning-rate 2e-3 --mod-a t1 --mod-b flair --model cycle --data-dir /media/hngu4068/8E50BB0C50BAF9D3/brats2018
