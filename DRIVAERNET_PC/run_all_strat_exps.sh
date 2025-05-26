#!/bin/bash
# Updated script for continual learning experiments with configurable LR scheduling
# Supports 3 scenarios: DrivAerNet+Bin, DrivAerNet++ Input, DrivAerNet++ Bin

BASE_SEED=42

# Configuration - modify these paths as needed
MODEL_PATH='models/PN_best.pth'
EPOCHS=200
NUM_ITERS=5
BATCH_SIZE=32

echo "Starting continual learning experiments with configurable LR scheduling..."
echo "Base seed: $BASE_SEED"
echo "Model path: $MODEL_PATH"
echo "Epochs: $EPOCHS, Iterations: $NUM_ITERS, Batch size: $BATCH_SIZE"
echo "==================================================================================="

# Scenario 1: DrivAerNet + Bin
echo "SCENARIO 1: DrivAerNet + Bin Experiments"
echo "========================================="

## Cumulative Experiments
echo "Running Cumulative strategy..."
python main.py \
    --exp_name 'DRIVAERNET-BIN_cumulative_regpointnetPT-20k_epochs200' \
    --strategy 'cumulative' \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Naive Experiments  
echo "Running Naive strategy..."
python main.py \
    --exp_name 'DRIVAERNET-BIN_naive_regpointnetPT-20k_epochs200' \
    --strategy 'naive' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Replay Experiments
echo "Running Replay strategy..."
python main.py \
    --exp_name 'DRIVAERNET-BIN_replay-membuff450_regpointnetPT-20k_epochs200' \
    --strategy 'replay' \
    --mem_size=450 \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## GEM Experiments
echo "Running GEM strategy..."
python main.py \
    --exp_name 'DRIVAERNET-BIN_gem-ppe75-ms2_regpointnetPT-20k_epochs200' \
    --strategy 'gem' \
    --gem_ppe=75 \
    --mem_strength=2 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## EWC Experiments
echo "Running EWC strategy..."
python main.py \
    --exp_name 'DRIVAERNET-BIN_ewc-lambda-1000_regpointnetPT-20k_epochs200' \
    --strategy 'ewc' \
    --ewc_lambda=1000 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

echo "SCENARIO 1 COMPLETED"
echo "==================================================================================="

# Scenario 2: DrivAerNet++ Input  
echo "SCENARIO 2: DrivAerNet++ Input Experiments"
echo "=========================================="

## Cumulative Experiments
echo "Running Cumulative strategy (++ Input)..."
python main.py \
    --exp_name 'DRIVAERNETplusINPUT_cumulative_regpointnetPT-20k_epochs200' \
    --strategy 'cumulative' \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Naive Experiments
echo "Running Naive strategy (++ Input)..."
python main.py \
    --exp_name 'DRIVAERNETplusINPUT_naive_regpointnetPT-20k_epochs200' \
    --strategy 'naive' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Replay Experiments
echo "Running Replay strategy (++ Input)..."
python main.py \
    --exp_name 'DRIVAERNETplusINPUT_replay-membuff450_regpointnetPT-20k_epochs200' \
    --strategy 'replay' \
    --mem_size=450 \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## GEM Experiments
echo "Running GEM strategy (++ Input)..."
python main.py \
    --exp_name 'DRIVAERNETplusINPUT_gem-ppe75-ms2_regpointnetPT-20k_epochs200' \
    --strategy 'gem' \
    --gem_ppe=75 \
    --mem_strength=2 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## EWC Experiments
echo "Running EWC strategy (++ Input)..."
python main.py \
    --exp_name 'DRIVAERNETplusINPUT_ewc-lambda-1000_regpointnetPT-20k_epochs200' \
    --strategy 'ewc' \
    --ewc_lambda=1000 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

echo "SCENARIO 2 COMPLETED"
echo "==================================================================================="

# Scenario 3: DrivAerNet++ Bin
echo "SCENARIO 3: DrivAerNet++ Bin Experiments"  
echo "========================================"

## Cumulative Experiments
echo "Running Cumulative strategy (++ Bin)..."
python main.py \
    --exp_name 'DRIVAERNETplusBIN_cumulative_regpointnetPT-20k_epochs200' \
    --strategy 'cumulative' \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Naive Experiments
echo "Running Naive strategy (++ Bin)..."
python main.py \
    --exp_name 'DRIVAERNETplusBIN_naive_regpointnetPT-20k_epochs200' \
    --strategy 'naive' \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## Replay Experiments
echo "Running Replay strategy (++ Bin)..."
python main.py \
    --exp_name 'DRIVAERNETplusBIN_replay-membuff450_regpointnetPT-20k_epochs200' \
    --strategy 'replay' \
    --mem_size=450 \
    --epochs=$EPOCHS \
    --start_iter 1 \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## GEM Experiments
echo "Running GEM strategy (++ Bin)..."
python main.py \
    --exp_name 'DRIVAERNETplusBIN_gem-ppe75-ms2_regpointnetPT-20k_epochs200' \
    --strategy 'gem' \
    --gem_ppe=75 \
    --mem_strength=2 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

## EWC Experiments
echo "Running EWC strategy (++ Bin)..."
python main.py \
    --exp_name 'DRIVAERNETplusBIN_ewc-lambda-1000_regpointnetPT-20k_epochs200' \
    --strategy 'ewc' \
    --ewc_lambda=1000 \
    --epochs=$EPOCHS \
    --num_iters $NUM_ITERS \
    --batch_size $BATCH_SIZE \
    --pretrained=True \
    --model_path=$MODEL_PATH \
    --seed $BASE_SEED

echo "SCENARIO 3 COMPLETED"
echo "==================================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "Results saved in results/ directory"
echo "Checkpoints saved in checkpoints/ directory"
echo "Logs saved in logs/ directory"