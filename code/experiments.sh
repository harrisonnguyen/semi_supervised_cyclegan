#! /bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/ssl_brats/semi_pair --batch-size 1 --n-epochs 50 --cycle-loss-weight 5.0 --summary-freq 100 --learning-rate 2e-3 --mod-a t1 --mod-b flair
