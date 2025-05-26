#!/bin/bash

## Naive Experiments
python main.py --exp_name 'SHAPENET-PC_naive_regpointnetPT-20k_epochs150' --strategy 'naive' --epochs=150 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth'

## Cumulative Experiments
python main.py --exp_name 'SHAPENET-PC_cumulative_regpointnetPT-20k_epochs150' --strategy 'cumulative' --epochs=150 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth'

## Replay Experiments
python main.py --exp_name 'SHAPENET-PC_replay-membuff1000_regpointnetPT-20k_epochs150' --strategy 'replay' --mem_size=1000 --epochs=150 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth' 

## EWC Experiments
python main.py --exp_name 'SHAPENET-PC_ewc-lambda-500_regpointnetPT-20k_epochs150' --strategy 'ewc' --ewc_lambda=500 --epochs=150 --lr 0.001 --num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth'

## GEM Experiments
python main.py --exp_name 'SHAPENET-PC_gem-ppe50-ms10_regpointnetPT-20k_epochs150' --strategy 'gem' --gem_ppe=50 --mem_strength=10 --epochs=150 ---lr 0.001 -num_iters 5 --batch_size 32 --pretrained=True --model_path='models/PN_best.pth'
