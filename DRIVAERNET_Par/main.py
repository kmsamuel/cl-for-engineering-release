#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: main.py

Unified main script for parametric DrivAerNet++ continual learning experiments.
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
import traceback
import random

# Your existing imports
from model import Regression_ResNet_Model
from datasets import create_parametric_dataset
from Parametric_Benchmark import (
    create_parametric_benchmark,
    get_parametric_bin_benchmark,
    get_parametric_input_benchmark
)

# Avalanche imports
from avalanche.evaluation.metrics import (
    regression_metrics, loss_metrics, forgetting_metrics,
    confusion_matrix_metrics, r2_metrics, cpu_usage_metrics, 
    disk_usage_metrics, timing_metrics
)
from avalanche.training.plugins import (
    ReplayPlugin, EWCPlugin, GEMPlugin, GEMPluginDRIVAERNET, 
    AGEMPlugin, LwFPlugin, EvaluationPlugin, RegressionMIRPlugin, 
    RARPlugin, GDumbPlugin
)
from avalanche.logging import InteractiveLogger, TextLogger

# Import your new configurable strategies
from avalanche.training.supervised import (
    Naive, Cumulative, create_strategy_with_lr_config, BenchmarkLRConfigs, RegressionDER, MER
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
        'xdim': 29,                     # Dimension of parametric design vector
        'ydim': 1,                      # trains regression model for each objective
        'tdim': args.network_dim,       # dimension of latent variable
        'net': [args.network_dim, args.network_dim],  # network architecture        
        'Training_Epochs': args.epochs, # number of training epochs
        'num_regressors': 1,            # number of regressors to train
        'Model_Labels': ['Regressor_Cd'], # labels for regressors
        'lr': args.lr,                  # learning rate
        'weight_decay': 0.0,            # weight decay
        'device_name': 'cuda',
        'multi_gpu': True
    }


def initialize_model(config, device, args):
    """Initialize the model based on configuration."""
    model = Regression_ResNet_Model(config)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    return model, optimizer


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


def get_benchmark(args):
    """Get the appropriate benchmark based on scenario."""
    if args.scenario == 'bin':
        return get_parametric_bin_benchmark()
    elif args.scenario == 'input':
        return get_parametric_input_benchmark()
    else:
        raise ValueError(f"Invalid scenario: {args.scenario}")


def create_cl_strategy(args, model, optimizer, criterion, eval_plugin, device):
    """Create the continual learning strategy with configurable LR."""
    
    # Get the LR configuration for parametric benchmark (reuse ShapeNet config)
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
            plugins=[ReplayPlugin(
                mem_size=args.mem_size, 
                batch_size=args.batch_size, 
                batch_size_mem=args.mem_batch_size
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'ewc':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[EWCPlugin(ewc_lambda=args.ewc_lambda)],
            **base_kwargs
        )
    
    elif args.strategy == 'gem':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[GEMPlugin(
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
    
    elif args.strategy == 'mir':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[RegressionMIRPlugin(
                mem_size=args.mem_size, 
                batch_size_mem=32, 
                subsample=300
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'rar':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[RARPlugin(
                mem_size=args.mem_size, 
                batch_size_mem=32
            )],
            **base_kwargs
        )
    
    elif args.strategy == 'gdumb':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[GDumbPlugin(mem_size=args.mem_size)],
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
    
    elif args.strategy == 'der':
        return RegressionDER(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=args.mem_size,
            batch_size_mem=32,
            **base_kwargs
        )
    
    elif args.strategy == 'mer':
        return MER(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=args.mem_size,
            batch_size_mem=32,
            n_inner_steps=5,
            beta=0.1,
            gamma=0.1,
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
        
        # Create model configuration
        config = create_model_config(args)
        
        # Initialize model
        print('=====> Building model...')
        model, optimizer = initialize_model(config, device, args)
        
        # Get benchmark
        benchmark = get_benchmark(args)
        
        # Setup logging
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{args.exp_name}_iter{iteration}_log.txt")
        
        # Determine number of classes based on scenario
        if args.scenario == 'bin':
            num_classes = 4
        else:  # input scenario
            num_classes = 3
        
        eval_plugin = EvaluationPlugin(
            regression_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            timing_metrics(epoch=True),
            cpu_usage_metrics(experience=True),
            confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
            r2_metrics(save_image=True, stream=True),
            disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            collect_all=True,
            loggers=[InteractiveLogger(), TextLogger(open(log_file, "w"))]
        )
        
        # Create criterion and strategy
        criterion = create_criterion(args.loss)
        cl_strategy = create_cl_strategy(args, model, optimizer, criterion, eval_plugin, device)
        
        print(f"Running {args.exp_name} - Iteration {iteration+1}")
        print(f"Scenario: {args.scenario}")
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
                num_workers=4,
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
    parser = argparse.ArgumentParser(description='Parametric DrivAerNet++ Continual Learning')
    
    # Scenario settings
    parser.add_argument('--scenario', type=str, default='bin',
                       choices=['bin', 'input'],
                       help='Incremental scenario (bin or input)')
    
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
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                       help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--loss', type=str, default='mse', choices=['l1', 'mse', 'gmean'],
                       help='Training loss type')
    parser.add_argument('--no_cuda', type=bool, default=False,
                       help='Enables CUDA training')
    parser.add_argument('--network_dim', type=int, default=348, 
                       help='Set neural network dimensions')
    
    # CL settings
    parser.add_argument('--strategy', type=str, default='naive', metavar='STRATEGY',
                       choices=['naive', 'cumulative', 'replay', 'ewc', 'lwf', 'gem', 'agem', 
                               'replay_ewc', 'gem_replay', 'mir', 'rar', 'gdumb', 'der', 'mer'],
                       help='Continual learning strategy')
    parser.add_argument('--mem_size', type=int, default=100, metavar='HYPERPARAM',
                       help='Size of memory buffer used in Replay')
    parser.add_argument('--mem_batch_size', type=int, default=16, metavar='HYPERPARAM',
                       help='Size of memory buffer batch used in Replay')
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
    
    # Additional parametric-specific settings
    parser.add_argument('--training_budget', type=int, default=815, metavar='HYPERPARAM',
                       help='Number of samples allowed to train during each exp')
    parser.add_argument('--sampling_temp', type=float, default=1.0, metavar='HYPERPARAM',
                       help='Boltzmann Sampling Temp')
    parser.add_argument('--n_clusters', type=int, default=4, metavar='HYPERPARAM',
                       help='Clusters to split into for Adaptive Replay')
    
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


# Example usage with the new unified system:
# Parametric bin incremental (4 classes):
# python main.py --scenario bin --exp_name 'Parametric_Bin_naive_test' --strategy 'naive' --epochs 100 --num_iters 1 --batch_size 256 --base_seed 42

# Parametric input incremental (3 classes):
# python main.py --scenario input --exp_name 'Parametric_Input_replay' --strategy 'replay' --mem_size 100 --epochs 100 --num_iters 4 --batch_size 256 --base_seed 42

# Advanced strategies:
# python main.py --scenario bin --exp_name 'Parametric_Bin_ewc' --strategy 'ewc' --ewc_lambda 0.01 --epochs 100 --batch_size 256 --base_seed 42
# python main.py --scenario bin --exp_name 'Parametric_Bin_replay_ewc' --strategy 'replay_ewc' --mem_size 100 --ewc_lambda 0.01 --epochs 100 --batch_size 256 --base_seed 42

    # python main.py --exp_name 'gem_m100' --strategy 'gem' --mem_size=100 --epochs=100 --num_iters 1 --batch_size 8 
    # python main.py --exp_name 'exp_test' --strategy 'naive' --mem_size=100 --epochs=100 --num_iters 1 --batch_size 32 
    # python main.py --exp_name 'DRIVAERNET-Par_basetest_bs32_200epochs_no-adapt-lr' --strategy 'naive' --mem_size=100 --epochs=100 --num_iters 1 --batch_size 32 