#!/bin/bash
BASE_SEED=42

## Cumulative Experiments
python main.py --exp_name 'SHIPDsub-Par_cumulative_resnet_epochs300' --strategy 'cumulative' --epochs=300 --num_iters 5 --batch_size 256 --lr=0.02 --base_seed $BASE_SEED

## Naive Experiments
python main.py --exp_name 'SHIPDsub-Par_naive_resnet_epochs300' --strategy 'naive' --epochs=300 --num_iters 5 --batch_size 256 --lr=0.02 --base_seed $BASE_SEED

# ## Replay Experiments
python main.py --exp_name 'SHIPDsub-Par_replay-membuff1600_resnet_epochs300' --strategy 'replay' --mem_size=1600 --epochs=300 --num_iters 5 --batch_size 256 --lr=0.02 --base_seed $BASE_SEED

## EWC Experiments
python main.py --exp_name 'SHIPDsub-Par_ewc-lambda-100_resnet_epochs300' --strategy 'ewc' --ewc_lambda=100 --epochs=150 --lr 0.001 --num_iters 5 --batch_size 256 --base_seed $BASE_SEED

## GEM Experiments
python main.py --exp_name 'SHIPDsub-Par_gem-ppe266-ms5_resnet_epochs300' --strategy 'gem' --gem_ppe=266 --mem_strength=5 --epochs=150 ---lr 0.001 -num_iters 5 --batch_size 256 --base_seed $BASE_SEED
