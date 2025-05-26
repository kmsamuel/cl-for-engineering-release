#!/bin/bash

BASE_SEED=42

## Replay Experiments
python main.py --exp_name 'RAADL-PC_replay-membuff105_regpointnetPT-20k_epochs200' --strategy 'replay' --mem_size=105 --epochs=200 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' --base_seed $BASE_SEED

## EWC Experiments
python main.py --exp_name 'RAADL-PC_ewc-lambda-1000_regpointnetPT-20k_epochs200' --strategy 'ewc' --ewc_lambda=1000 --epochs=200 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' --base_seed $BASE_SEED


## Cumulative Experiments
python main.py --exp_name 'RAADL-PC_cumulative_regpointnetPT-20k_epochs200' --strategy 'cumulative' --epochs=200 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' --base_seed $BASE_SEED

## GEM Experiments

python main.py --exp_name 'RAADL-PC_gem-ppe27-ms0-25_regpointnetPT-20k_epochs200' --strategy 'gem' --gem_ppe=27 --mem_strength=0.25 --epochs=200 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' --base_seed $BASE_SEED


## Naive Experiments
python main.py --exp_name 'RAADL-PC_naive_regpointnetPT-20k_epochs200' --strategy 'naive' --epochs=200 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' --base_seed $BASE_SEED
