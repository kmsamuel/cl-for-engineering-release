#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: main.py

Unified main script for SHIPD ship hydrodynamics continual learning experiments.
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
from datasets import create_shipd_dataset
from SHIPD_Benchmark import (
    create_shipd_benchmark,
    get_shipd_benchmark
)

# Avalanche imports
from avalanche.evaluation.metrics import (
    regression_metrics, loss_metrics, forgetting_metrics,
    confusion_matrix_metrics, r2_metrics, cpu_usage_metrics, 
    disk_usage_metrics, timing_metrics, MAC_metrics
)
from avalanche.training.plugins import (
    ReplayPlugin, EWCPlugin, GEMPlugin, GEMPluginDRIVAERNET, 
    AGEMPlugin, LwFPlugin, EvaluationPlugin, RegressionMIRPlugin, 
    RARPlugin, GDumbPlugin, TabPFNAdaptiveReplayPlugin, 
    KMeansInputAdaptiveReplayPlugin, ExperienceAdaptiveReplayPlugin
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
        'xdim': 44,                     # Dimension of ship design vector
        'ydim': 1,                      # trains regression model for each objective
        'tdim': 256,                    # dimension of latent variable
        'net': [256, 256],              # network architecture        
        'Training_Epochs': args.epochs, # number of training epochs
        'num_regressors': 1,            # number of regressors to train
        'Model_Labels': ['Regressor_Cw'], # labels for regressors
        'lr': args.lr,                  # learning rate
        'weight_decay': 3e-6,           # weight decay
        'device_name': 'cuda',
        'multi_gpu': True
    }


def initialize_model(config, device, args):
    """Initialize the model based on configuration."""
    model = Regression_ResNet_Model(config)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print('Using 2 GPUs')
        
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=config['weight_decay']
    )
    
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
    """Get the SHIPD benchmark."""
    return get_shipd_benchmark()


def create_cl_strategy(args, model, optimizer, criterion, eval_plugin, device):
    """Create the continual learning strategy with configurable LR."""
    
    # Get the LR configuration for SHIPD benchmark (reuse ShapeNet config)
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
    
    elif args.strategy == 'tabpfn_adaptive_replay':
        visualization_path = os.path.join(f'results/embedding_vis/iter_{args.current_iter}', args.exp_name)
        os.makedirs(visualization_path, exist_ok=True)

        plugin = TabPFNAdaptiveReplayPlugin(
            mem_size=args.mem_size,
            batch_size=args.batch_size,
            batch_size_mem=args.mem_batch_size,
            temperature=args.sampling_temp,
            vis_path=visualization_path
        )
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[plugin],
            **base_kwargs
        )

    elif args.strategy == 'adaptive_replay_param':
        visualization_path = os.path.join(f'results/param_cluster_vis/iter_{args.current_iter}', args.exp_name)
        os.makedirs(visualization_path, exist_ok=True)

        plugin = KMeansInputAdaptiveReplayPlugin(
            mem_size=args.mem_size,
            batch_size=args.batch_size,
            batch_size_mem=args.mem_batch_size,
            temperature=args.sampling_temp,
            vis_path=visualization_path
        )
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[plugin],
            **base_kwargs
        )
        
    elif args.strategy == 'adaptive_replay_expclusters':
        visualization_path = os.path.join(f'results/param_cluster_vis/iter_{args.current_iter}', args.exp_name)
        os.makedirs(visualization_path, exist_ok=True)

        plugin = ExperienceAdaptiveReplayPlugin(
            mem_size=args.mem_size,
            batch_size=args.batch_size,
            batch_size_mem=args.mem_batch_size,
            temperature=args.sampling_temp,
            vis_path=visualization_path
        )
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=[plugin],
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
        
        # Store current iteration for visualization paths
        args.current_iter = iteration
        
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
        
        eval_plugin = EvaluationPlugin(
            regression_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            timing_metrics(epoch=True),
            MAC_metrics(epoch=True, experience=True),
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
            
            cl_strategy.train(
                experience,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
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
    parser = argparse.ArgumentParser(description='SHIPD Ship Hydrodynamics Continual Learning')
    
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
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'mse', 'gmean'],
                       help='Training loss type')
    parser.add_argument('--no_cuda', type=bool, default=False,
                       help='Enables CUDA training')
    
    # CL settings
    parser.add_argument('--strategy', type=str, default='naive', metavar='STRATEGY',
                       choices=['naive', 'cumulative', 'replay', 'ewc', 'lwf', 'gem', 'agem', 
                               'tabpfn_adaptive_replay', 'adaptive_replay_param', 
                               'adaptive_replay_expclusters'],
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
    
    # Advanced replay settings
    parser.add_argument('--sampling_temp', type=float, default=1.0, metavar='HYPERPARAM',
                       help='Boltzmann Sampling Temperature')
    parser.add_argument('--n_clusters', type=int, default=4, metavar='HYPERPARAM',
                       help='Clusters to split into for Adaptive Replay')
    
    # Model-specific settings from original
    parser.add_argument('--lds', action='store_true', default=False, 
                       help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                       choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, 
                       help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, 
                       help='LDS gaussian/laplace kernel sigma')
    parser.add_argument('--fds', action='store_true', default=False, 
                       help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                       choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=9, 
                       help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=1, 
                       help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, 
                       help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, 
                       help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=100, 
                       help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                       help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
    
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
# Basic strategies:
# python main.py --exp_name 'SHIPD_naive_test' --strategy 'naive' --epochs 100 --num_iters 1 --batch_size 256 --base_seed 42

# Replay strategies:
# python main.py --exp_name 'SHIPD_replay' --strategy 'replay' --mem_size 100 --epochs 100 --num_iters 4 --batch_size 256 --base_seed 42

# Advanced adaptive replay:
# python main.py --exp_name 'SHIPD_tabpfn_adaptive' --strategy 'tabpfn_adaptive_replay' --mem_size 100 --sampling_temp 1.0 --epochs 100 --batch_size 256 --base_seed 42

# EWC strategy:
# python main.py --exp_name 'SHIPD_ewc' --strategy 'ewc' --ewc_lambda 0.001 --epochs 100 --batch_size 256 --base_seed 42
    
    # python main.py --exp_name 'gem_m100' --strategy 'gem' --mem_size=100 --epochs=100 --num_iters 1 --batch_size 8
    # python main.py --exp_name 'SHIPD-Par_naive_resnet_epochs100' --strategy 'naive' --mem_size=100 --epochs=100 --num_iters 1 --batch_size 32