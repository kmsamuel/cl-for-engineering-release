#!/bin/bash
# DrivAerNet Parametric continual learning experiments
# Two scenarios: DrivAerNet-Par+Bin and DrivAerNet-Par+Input

BASE_SEED=42
EPOCHS=150
NUM_ITERS=5
BATCH_SIZE=32

echo "Starting DrivAerNet Parametric continual learning experiments..."
echo "Base seed: $BASE_SEED"
echo "Epochs: $EPOCHS, Iterations: $NUM_ITERS, Batch size: $BATCH_SIZE"
echo "==================================================================================="

# Scenario 1: DrivAerNet-Par + Bin
echo "SCENARIO 1: DrivAerNet-Par + Bin Experiments"
echo "============================================="

## Cumulative Experiments
echo "Running Cumulative strategy (Par+Bin)..."
python main.py \
    --exp_name 'DRIVAERNET-ParBIN_cumulative_resnet_epochs150' \
    --strategy 'cumulative' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## Naive Experiments
echo "Running Naive strategy (Par+Bin)..."
python main.py \
    --exp_name 'DRIVAERNET-ParBIN_naive_resnet_epochs150' \
    --strategy 'naive' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## Replay Experiments
echo "Running Replay strategy (Par+Bin)..."
python main.py \
    --exp_name 'DRIVAERNET-ParBIN_replay-membuff450_resnet_epochs150' \
    --strategy 'replay' \
    --mem_size=550 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## GEM Experiments
echo "Running GEM strategy (Par+Bin)..."
python main.py \
    --exp_name 'DRIVAERNET-ParBIN_gem-ppe190-ms5_resnet_epochs150' \
    --strategy 'gem' \
    --gem_ppe=190 \
    --mem_strength=5 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## EWC Experiments
echo "Running EWC strategy (Par+Bin)..."
python main.py \
    --exp_name 'DRIVAERNET-ParBIN_ewc-lambda-15000_resnet_epochs150' \
    --strategy 'ewc' \
    --ewc_lambda=15000 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

echo "SCENARIO 1 COMPLETED"
echo "==================================================================================="

# Scenario 2: DrivAerNet-Par + Input
echo "SCENARIO 2: DrivAerNet-Par + Input Experiments"
echo "==============================================="

## Cumulative Experiments
echo "Running Cumulative strategy (Par+Input)..."
python main.py \
    --exp_name 'DRIVAERNET-ParINPUT_cumulative_resnet_epochs150' \
    --strategy 'cumulative' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## Naive Experiments
echo "Running Naive strategy (Par+Input)..."
python main.py \
    --exp_name 'DRIVAERNET-ParINPUT_naive_resnet_epochs150' \
    --strategy 'naive' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## Replay Experiments
echo "Running Replay strategy (Par+Input)..."
python main.py \
    --exp_name 'DRIVAERNET-ParINPUT_replay-membuff450_resnet_epochs150' \
    --strategy 'replay' \
    --mem_size=550 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## GEM Experiments
echo "Running GEM strategy (Par+Input)..."
python main.py \
    --exp_name 'DRIVAERNET-ParINPUT_gem-ppe190-ms5_resnet_epochs150' \
    --strategy 'gem' \
    --gem_ppe=190 \
    --mem_strength=5 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

## EWC Experiments
echo "Running EWC strategy (Par+Input)..."
python main.py \
    --exp_name 'DRIVAERNET-ParINPUT_ewc-lambda-15000_resnet_epochs150' \
    --strategy 'ewc' \
    --ewc_lambda=15000 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --seed $BASE_SEED

echo "SCENARIO 2 COMPLETED"
echo "==================================================================================="
echo "ALL DRIVAERNET PARAMETRIC EXPERIMENTS COMPLETED!"
echo "Results saved in results/ directory"
echo "Checkpoints saved in checkpoints/ directory"
echo "Logs saved in logs/ directory"