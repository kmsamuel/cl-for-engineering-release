#!/bin/bash
BASE_SEED=42

# Cumulative Experiments
python main.py --exp_name 'SHIPD-PC_cumulative_regpointnetPT-20k_epochs150' --strategy 'cumulative' --epochs=150 --lr 0.0005 --num_iters 5 --batch_size 80 --loss 'mse' --pretrained='PointNet' --model_path='models/PN_best.pth' --base_seed $BASE_SEED

# Naive Experiments
python main.py --exp_name 'SHIPD-PC_naive_regpointnetPT-20k_epochs150' --strategy 'naive' --epochs=150 --lr 0.0005 --num_iters 5 --batch_size 80 --pretrained='PointNet' --model_path='models/PN_best.pth' --loss 'mse' --base_seed $BASE_SEED

# ## GEM Experiments
python main.py --exp_name 'SHIPD-PC_gem-ppe80-ms2_regpointnetPT-20k_epochs150' --strategy 'gem' --gem_ppe=80 --mem_strength=2 --epochs=150 --lr 0.0005 --num_iters 5 --batch_size 80 --pretrained='PointNet' --model_path='models/PN_best.pth' --base_seed $BASE_SEED

## Replay Experiments
python main.py --exp_name 'SHIPD-PC_replay-membuff1600_regpointnetPT-20k_epochs150' --strategy 'replay' --mem_size=1600 --epochs=150 --lr 0.0005 --num_iters 5  --batch_size 32 --loss 'mse' --pretrained='PointNet' --model_path='models/PN_best.pth' --base_seed $BASE_SEED

# ## EWC Experiments
python main.py --exp_name 'SHIPD-PC_ewc-lambda-10000_regpointnetPT-20k_epochs10' --strategy 'ewc' --ewc_lambda=10000 --epochs=150 --lr 0.0005 --num_iters 5 --batch_size 80 --pretrained='PointNet' --model_path='models/PN_best.pth' --base_seed $BASE_SEED



