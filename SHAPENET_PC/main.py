#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: main.py

Clean main script using configurable LR scheduling.
"""

from __future__ import print_function
import os
import argparse
import torch
from io import BytesIO
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import traceback
import random

# Your existing imports
from model import RegDGCNN, RegPointNet
from pretrained_model import RegPointNetPT
from datasets import ShapeNetCars
from ShapeNetCars_Benchmark import ShapeNetCars_Benchmark

# Avalanche imports - using your modified library
from avalanche.evaluation.metrics import (
    regression_metrics, loss_metrics, forgetting_metrics,
    confusion_matrix_metrics, r2_metrics, cpu_usage_metrics, 
    disk_usage_metrics, timing_metrics
)
from avalanche.training.plugins import (
    ReplayPlugin, EWCPlugin, GEMPlugin, GEMPluginDRIVAERNET, 
    AGEMPlugin, LwFPlugin, EvaluationPlugin
)
from avalanche.logging import InteractiveLogger, TextLogger

# Import your new configurable strategies
from avalanche.training.supervised import (
    Naive, Cumulative, create_strategy_with_lr_config, BenchmarkLRConfigs
)

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def set_seed(seed=24):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _init_(args):
    """Initialize directory structure."""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    if not os.path.exists(f'checkpoints/{args.exp_name}'):
        os.makedirs(f'checkpoints/{args.exp_name}')
    
    for i in range(args.num_iters):
        iter_dir = f'checkpoints/{args.exp_name}/iter_{i+1}'
        models_dir = f'{iter_dir}/models'

        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Back up main.py to each iteration folder
        os.system(f'cp main.py {iter_dir}/main_reg.py.backup')


def create_model_config(args):
    """Create model configuration dictionary."""
    return {
        'cuda': True,
        'seed': args.seed,
        'num_points': 20000,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'emb_dims': 512,
        'k': 40,
        'optimizer': 'adam',
        'output_channels': 1
    }


def initialize_model(config, device, args):
    """Initialize the model based on configuration."""
    if args.pretrained:
        model = load_pretrained_RegPNmodel_and_unfreeze_layers(
            args.model_path, device, args.unfreeze_stages
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr
        )
    else:
        model = RegPointNet(args=config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Enable DataParallel if multiple GPUs available
    if config['cuda'] and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print(f'Using {torch.cuda.device_count()} GPUs')
    
    return model, optimizer


def load_pretrained_RegPNmodel_and_unfreeze_layers(model_path, device, unfreeze_stages=None):
    """Load a saved model and return the model with the loaded weights."""
    
    # Load the file into a buffer
    with open(model_path, "rb") as f:
        buffer = BytesIO(f.read())

    # Now load the checkpoint from the buffer
    checkpoint = torch.load(buffer, map_location=device)
    model_state_dict = checkpoint['model']

    # Initialize your model
    model = RegPointNetPT().to(device)
    
    # Use DataParallel if multiple GPUs are available and specified in config
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print('Using 2 GPUs')

    # Check if "module." prefix exists in loaded model and adjust if necessary
    new_state_dict = {}
    if any(key.startswith("module.") for key in model_state_dict.keys()):
        new_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
    else:
        new_state_dict = {"module." + key: value for key, value in model_state_dict.items()}
        
    # Load the adjusted state dictionary
    model.load_state_dict(new_state_dict, strict=False)
    
    # Step 2: Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: Unfreeze selected layers
    # If no stages specified, default to unfreezing last layer only
    if unfreeze_stages is None:
        unfreeze_stages = ['prediction']  # default to unfreezing final prediction head
    
    # Unfreeze specified layers
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Access the underlying model

    for name, module in model.named_children():
        if name == "encoder":
            for enc_name, enc_layer in module.named_children():
                if enc_name in unfreeze_stages:  # Check if this layer should be unfrozen
                    for param in enc_layer.parameters():
                        param.requires_grad = True

        elif name == "prediction":
            for pred_name, pred_layer in module.named_children():
                if pred_name in unfreeze_stages:  # Check if this layer should be unfrozen
                    for param in pred_layer.parameters():
                        param.requires_grad = True

    return model


def create_criterion(loss_type):
    """Create loss criterion based on type."""
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss(reduction='mean')
    elif loss_type == 'gmean':
        return nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_cl_strategy(args, model, optimizer, criterion, eval_plugin, device):
    """Create the continual learning strategy with configurable LR."""
    
    # Get the LR configuration for ShapeNet benchmark
    lr_config = BenchmarkLRConfigs.shapenet_cars()
    
    base_kwargs = {
        'train_mb_size': args.batch_size,
        'train_epochs': args.epochs,
        'eval_mb_size': args.batch_size,
        'evaluator': eval_plugin,
        'device': device,
        'lr_config': lr_config  # Pass the LR config
    }
    
    if args.strategy == 'naive':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **base_kwargs
        )
    
    elif args.strategy == 'cumulative':
        return Cumulative(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            **base_kwargs
        )
    
    elif args.strategy == 'replay':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[ReplayPlugin(mem_size=args.mem_size)],
            **base_kwargs
        )
    
    elif args.strategy == 'ewc':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[EWCPlugin(
                ewc_lambda=args.ewc_lambda, 
                mode="online", 
                decay_factor=0.9, 
                keep_importance_data=True
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'gem':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[GEMPluginDRIVAERNET(
                patterns_per_experience=args.gem_ppe,
                memory_strength=args.mem_strength
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'agem':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[AGEMPlugin(
                patterns_per_experience=args.gem_ppe,
                sample_size=args.sample_size
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'lwf':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[LwFPlugin(
                alpha=args.lwf_alpha,
                temperature=args.lwf_temp
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'replay_ewc':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[
                ReplayPlugin(mem_size=args.mem_size), 
                EWCPlugin(ewc_lambda=args.ewc_lambda)
            ],
            **base_kwargs
        )
    
    elif args.strategy == 'gem_replay':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[
                ReplayPlugin(mem_size=args.mem_size), 
                GEMPlugin(patterns_per_experience=args.gem_ppe, memory_strength=args.mem_strength)
            ],
            **base_kwargs
        )
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def run_single_iteration(args, io, iteration):
    """Run a single iteration of the experiment."""
    try:
        # Set seed for this iteration
        current_seed = args.base_seed + iteration
        set_seed(current_seed)
        print(f"Iteration {iteration+1}/{args.num_iters} with seed {current_seed}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Create model configuration
        config = create_model_config(args)
        
        # Initialize model
        print('=====> Building model...')
        model, optimizer = initialize_model(config, device, args)
        
        # Get benchmark
        benchmark = ShapeNetCars_Benchmark
        
        # Setup logging
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{args.exp_name}_iter{iteration}_log.txt")
        
        eval_plugin = EvaluationPlugin(
            regression_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            timing_metrics(epoch=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=4, save_image=False, stream=True),
            r2_metrics(save_image=True, stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            collect_all=True,
            loggers=[InteractiveLogger(), TextLogger(open(log_file, "w"))]
        )
        
        # Create criterion and strategy
        criterion = create_criterion(args.loss)
        cl_strategy = create_cl_strategy(args, model, optimizer, criterion, eval_plugin, device)
        
        print(f"Running {args.exp_name} - Iteration {iteration+1}")
        start_time = time.time()
        
        # Training loop
        print('Starting experiment...')
        results = []
        class_order = []
        
        for experience in benchmark.train_stream:
            print("Start of experience ", experience.current_experience)
            class_order.append(experience.classes_in_this_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            
            cl_strategy.train(
                experience,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True
            )
            print('Training completed')
            
            print('Computing accuracy on the whole test set')
            results.append(cl_strategy.eval(benchmark.test_stream))
        
        end_time = time.time()
        
        # Process and save results
        process_results(args, results, iteration, io, start_time, end_time, class_order, cl_strategy)
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred in iteration {iteration}: {e}")
        print(f"Traceback:\n{tb}")


def process_results(args, results, iteration, io, start_time, end_time, class_order, cl_strategy):
    """Process and save experiment results."""
    all_mae = []
    all_r2 = []
    
    # Save plots
    result_directory = f'results/{args.exp_name}/figs'
    iter_directory = os.path.join(result_directory, f'iter_{iteration}')
    os.makedirs(iter_directory, exist_ok=True)
    
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        filename = os.path.join(iter_directory, f'results_plots_exp{fig_num-1}.png')
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    
    # Extract metrics
    for result_dict in results:
        mae_metrics = result_dict['MAE_Stream/eval_phase/test_stream']
        r2_vals = result_dict['R2_Stream/eval_phase/test_stream']
        all_mae.append(mae_metrics)
        all_r2.append(r2_vals)
    
    # Save best model
    if all_mae:
        test_mae = min(all_mae)[0]
        model_path = f'checkpoints/{args.exp_name}/iter_{iteration+1}/models/model_{args.strategy}.pt'
        torch.save(cl_strategy.model.state_dict(), model_path)
        io.cprint(f'Saving model iteration {iteration}')
    
    # Log results
    io.cprint(f'MAE: {all_mae}')
    io.cprint(f'R2: {all_r2}')
    io.cprint(f'Class intro order: {class_order}')
    
    # Save results to files
    exp_name = args.exp_name
    os.makedirs(f'results/{exp_name}', exist_ok=True)
    
    with open(f'results/{exp_name}/mae.txt', 'a') as file:
        file.write(f"{all_mae}\n")
    
    with open(f'results/{exp_name}/r2.txt', 'a') as file:
        file.write(f"{all_r2}\n")
    
    elapsed_time = end_time - start_time
    with open(f'results/{exp_name}/time.txt', 'a') as file:
        file.write(f"{elapsed_time:.2f}\n")
    
    io.cprint(f"Elapsed time for iteration {iteration+1}: {elapsed_time:.2f} seconds")


def CLtrain(args, io):
    """Main training function."""
    best_mae = 100
    
    # Run iterations
    for j in range(args.start_iter, args.num_iters):
        run_single_iteration(args, io, j)
    
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='exp', metavar='NAME',
                       help='Name of the experiment')
    parser.add_argument('--num_iters', type=int, default=5,
                       help='Number of iterations')
    parser.add_argument('--start_iter', type=int, default=0,
                       help='Specify first iteration')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                       help='Random seed (default: 1)')
    parser.add_argument('--base_seed', type=int, default=42, 
                       help='Base random seed for iterations')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                       help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--loss', type=str, default='mse', choices=['l1', 'mse', 'gmean'],
                       help='Training loss type')
    parser.add_argument('--no_cuda', type=bool, default=False,
                       help='Enables CUDA training')
    
    # CL settings
    parser.add_argument('--strategy', type=str, default='naive', metavar='STRATEGY',
                       choices=['naive', 'cumulative', 'replay', 'ewc', 'lwf', 'gem', 'agem', 'replay_ewc', 'gem_replay'],
                       help='Continual learning strategy')
    parser.add_argument('--mem_size', type=int, default=100, metavar='HYPERPARAM',
                       help='Size of memory buffer used in Replay')
    parser.add_argument('--ewc_lambda', type=float, default=0.001, metavar='HYPERPARAM',
                       help='Lambda used for EWC')
    parser.add_argument('--gem_ppe', type=int, default=100, metavar='HYPERPARAM',
                       help='Patterns per experience GEM/AGEM')
    parser.add_argument('--mem_strength', type=float, default=0.5, metavar='HYPERPARAM',
                       help='Strength of memory buffer GEM')
    parser.add_argument('--sample_size', type=int, default=10, metavar='HYPERPARAM',
                       help='Number of patterns in memory sample when computing reference gradient AGEM')
    parser.add_argument('--lwf_alpha', type=int, default=1, metavar='HYPERPARAM',
                       help='Distillation hyperparameter LwF')
    parser.add_argument('--lwf_temp', type=int, default=2, metavar='HYPERPARAM',
                       help='Softmax temperature for distillation LwF')
    
    # Pretrained model settings
    parser.add_argument('--model_path', type=str, default='', metavar='PATH',
                       help='Pretrained model path')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='BOOL',
                       help='True if using pretrained model')
    parser.add_argument('--unfreeze_stages', nargs='+', default=['stn', 'fstn', 'head'],
                       help='Stages to unfreeze in pretrained model')
    
    args = parser.parse_args()
    
    # Initialize directories
    _init_(args)
    
    # Setup logging
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        io.cprint(f'Using GPU : {str(torch.cuda.current_device())} from {str(torch.cuda.device_count())} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    
    # Run training
    CLtrain(args, io)
    io.close()


# Example usage with the new clean system:
# python main.py --exp_name 'ShapeNetCars_clean_test' --strategy 'naive' --epochs 2 --num_iters 1 --batch_size 8 --base_seed 42
# python main.py --exp_name 'ShapeNetCars_replay_clean' --strategy 'replay' --mem_size 1000 --epochs 200 --num_iters 4 --batch_size 8 --base_seed 42
# python main.py --exp_name 'ShapeNetCars_ewc_clean' --strategy 'ewc' --ewc_lambda 0.01 --epochs 150 --batch_size 12 --base_seed 42

# Example usage
    # python main.py --exp_name 'ShapeNetCars_replay_pn20k_e200' --strategy 'replay' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8 
    # python main.py --exp_name 'ShapeNetCars_replay_pn20k_e200' --strategy 'gem' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8
    # python main.py --exp_name 'ShapeNetCars_debug' --strategy 'naive' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 12
    
    # Pretrained model
    # python main.py --exp_name 'ShapeNetCars_PT_baseline_pn20k_e100' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=100 --num_iters 1 --batch_size 8 
    # python main.py --exp_name 'ShapeNetCars_PT_exptest' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=1 --num_iters 1 --batch_size 8
    # python main.py --exp_name 'SHAPENET-PC_baseline_regpointnetPT-20k_epochs150_lr0-001-adapt' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=150 --num_iters 1 --batch_size 32

