#!/usr/bin/env bash
#train
python train.py --model ed-former --dataset RESIDE-IN --exp indoor --lambda_l1_start 0.1 --lambda_l1_end 1
python train.py --model ed-former --dataset RESIDE-OUT --exp outdoor --lambda_l1_start 0.5 --lambda_l1_end 0.9
python train.py --model ed-former --dataset RSHaze --exp rshaze --lambda_l1_start 0.1 --lambda_l1_end 1

#test
python test.py --model ed-former --dataset RESIDE-IN --exp indoor
python test.py --model ed-former --dataset RESIDE-OUT --exp outdoor
python test.py --model ed-former --dataset RSHaze --exp rshaze

