#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAADL Point Cloud Continual Learning Training Script

Main training script for continual learning experiments on the RAADL point cloud dataset.
Supports various continual learning strategies and neural network architectures for 
airplane aerodynamic coefficient prediction.

Author: Kaira Samuel
Contact: kmsamuel@mit.edu
"""

from __future__ import print_function
import os
import argparse
import torch
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import time
import torch.backends.cudnn as cudnn
import random
import traceback
from itertools import chain
from torch.optim import SGD

# Local imports
from pretrained_model_fltconds import RegPointNetPT
from datasets import RAADLPointCloudDataset
from RAADL_Benchmark import create_raadl_benchmark

# Avalanche imports - using your modified library with LR configs
from avalanche.training.supervised import (
    Naive, Cumulative, create_strategy_with_lr_config, BenchmarkLRConfigs
)
from avalanche.evaluation.metrics import (
    regression_metrics, loss_metrics, forgetting_metrics, bwt_metrics,
    confusion_matrix_metrics, r2_metrics, cpu_usage_metrics, 
    disk_usage_metrics, gpu_usage_metrics, MAC_metrics, 
    ram_usage_metrics, timing_metrics
)
from avalanche.training.plugins import (
    ReplayPlugin, EWCPlugin, GEMPlugin, GEMPluginDRIVAERNET, 
    AGEMPlugin, LwFPlugin, GenerativeReplayPlugin, EvaluationPlugin, 
    LRSchedulerPlugin, SupervisedPlugin
)
from avalanche.logging import InteractiveLogger, TextLogger

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

# Clear GPU cache
torch.cuda.empty_cache()


class IOStream():
    """Simple logging utility that writes to both console and file."""
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    """Initialize experiment directories."""
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    if not os.path.exists(f'checkpoints/{args.exp_name}'):
        os.makedirs(f'checkpoints/{args.exp_name}')
    
    for i in range(args.num_iters):
        iter_dir = f'checkpoints/{args.exp_name}/iter_{i+1}'
        models_dir = f'{iter_dir}/models'

        # Create directories for each iteration
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Back up main.py to each iteration folder
        os.system(f'cp main.py {iter_dir}/main_reg.py.backup')


def set_seed(seed=24):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained_RegPNmodel_and_unfreeze_layers(model_path, device, unfreeze_stages=None):
    """Load a saved model and return the model with the loaded weights."""
    
    # Load the file into a buffer
    with open(model_path, "rb") as f:
        buffer = BytesIO(f.read())

    # Now load the checkpoint from the buffer
    checkpoint = torch.load(buffer, map_location=device)
    model_state_dict = checkpoint['model']

    # Initialize your model
    model = RegPointNetPT(output_channels=1, condition_dim=3).to(device)
    
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

    def init_new_layers(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    # Initialize new layers before freezing
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
        
    base_model.prediction.fc_conditions.apply(init_new_layers)
    base_model.prediction.final_layer.apply(init_new_layers)

    # Step 2: Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Ensure new layers are always unfrozen
    for param in base_model.prediction.fc_conditions.parameters():
        param.requires_grad = True
    for param in base_model.prediction.final_layer.parameters():
        param.requires_grad = True  
        
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


def CLtrain(args, io):
    """Main continual learning training function."""
    best_mae = 100
    
    for j in range(args.start_iter, args.num_iters):
        try:
            config = {
                'exp_name': 'CdPrediction_ShapeNet_r2_100epochs_5k',
                'cuda': True,
                'seed': 1,
                'num_points': 20000,
                'lr': 0.001,
                'batch_size': 7,
                'epochs': 100,
                'dropout': 0.4,
                'emb_dims': 512,
                'k': 40,
                'optimizer': 'adam',
                'output_channels': 1
            }
            current_seed = args.base_seed + j
            set_seed(current_seed)
            print(f"Iteration {j+1}/{args.num_iters} with seed {current_seed}")
            
            # Set the device for training
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")

            print('=====> Building model...')
            
            if args.pretrained:
                new_model = load_pretrained_RegPNmodel_and_unfreeze_layers(args.model_path, device=device, unfreeze_stages=['stn', 'fstn', 'head'])
                print('LOADED PRETRAINED MODEL')
                
                def print_frozen_layers(model):
                    for name, param in model.named_parameters():
                        print(f"{name}: requires_grad = {param.requires_grad}")

                # Call this after loading
                print_frozen_layers(new_model)
                
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, new_model.parameters()))
            else:
                print('LOADED INITIALIZED MODEL')
                # Note: initialize_model function not implemented for RAADL
                raise NotImplementedError("Training from scratch not implemented for RAADL")

            # Create benchmark
            benchmark = create_raadl_benchmark()
            
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            log_file = os.path.join(log_dir, f"{args.exp_name}_iter{j}_log.txt")        
                
            eval_plugin = EvaluationPlugin(
                regression_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                forgetting_metrics(experience=True, stream=True),
                timing_metrics(epoch=True),
                cpu_usage_metrics(experience=True),
                r2_metrics(save_image=True, stream=True, multi_dims=False),
                disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                collect_all=True,
                loggers=[InteractiveLogger(), TextLogger(open(log_file, "w"))]
            )

            # Create loss criterion
            if args.loss == 'l1':
                criterion = nn.L1Loss()
            elif args.loss == 'mse':
                criterion = nn.MSELoss(reduction='mean')
            elif args.loss == 'gmean':
                criterion = nn.L1Loss(reduction='none')
            else:
                raise ValueError(f"Unsupported loss type: {args.loss}")

            # Create LR configuration for RAADL benchmark
            try:
                # Try to use RAADL-specific config if available
                lr_config = BenchmarkLRConfigs.raadl_pc()
            except AttributeError:
                # Fallback to ShapeNet config if RAADL config not available
                lr_config = BenchmarkLRConfigs.shapenet_cars()
                
            # CHOOSE THE STRATEGY INSTANCE
            base_strategy_kwargs = {
                'model': new_model,
                'optimizer': torch.optim.Adam(new_model.parameters(), lr=args.lr),
                'criterion': criterion,
                'train_mb_size': args.batch_size,
                'train_epochs': args.epochs,
                'eval_mb_size': args.batch_size,
                'evaluator': eval_plugin,
                'lr_config': lr_config,  # Add LR config
                'device': device
            }
            
            if args.strategy == 'naive':
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'cumulative':
                cl_strategy = Cumulative(**base_strategy_kwargs)
                
            elif args.strategy == 'replay':
                base_strategy_kwargs['plugins'] = [ReplayPlugin(mem_size=args.mem_size)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'ewc':
                base_strategy_kwargs['plugins'] = [EWCPlugin(ewc_lambda=args.ewc_lambda, mode="online", decay_factor=0.9, keep_importance_data=True)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'gem':
                base_strategy_kwargs['plugins'] = [GEMPluginDRIVAERNET(patterns_per_experience=args.gem_ppe, memory_strength=args.mem_strength)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'agem':
                base_strategy_kwargs['plugins'] = [AGEMPlugin(patterns_per_experience=args.gem_ppe, sample_size=args.sample_size)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'lwf':
                base_strategy_kwargs['plugins'] = [LwFPlugin(alpha=args.lwf_alpha, temperature=args.lwf_temp)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'replay_ewc':
                base_strategy_kwargs['plugins'] = [ReplayPlugin(mem_size=args.mem_size), EWCPlugin(ewc_lambda=args.ewc_lambda)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            elif args.strategy == 'gem_replay':
                base_strategy_kwargs['plugins'] = [ReplayPlugin(mem_size=args.mem_size), GEMPlugin(patterns_per_experience=args.gem_ppe, memory_strength=args.mem_strength)]
                cl_strategy = Naive(**base_strategy_kwargs)
                
            else:
                raise ValueError(f"Unsupported strategy: {args.strategy}")
                               
            print(f"Running {args.exp_name} - Iteration {j+1}")
            start_time = time.time()
            
            # TRAINING LOOP
            print('Starting experiment...')
            results = []
            class_order = []
            
            for experience in benchmark.train_stream:
                print("Start of experience ", experience.current_experience)
                
                class_order.append(experience.classes_in_this_experience)
                print("Current Classes: ", experience.classes_in_this_experience)
                
                cl_strategy.train(
                    experience,
                    num_workers=16,
                    pin_memory=True,
                    persistent_workers=True,
                    drop_last=True
                )
                print('Training completed')

                print('Computing accuracy on the whole test set')
                results.append(cl_strategy.eval(benchmark.test_stream))
                fin_model = cl_strategy.model.state_dict()
                
            end_time = time.time()
            
            # Process results
            all_mae = []
            all_r2 = []
            
            result_directory = f'results/{args.exp_name}/figs'
            iter_directory = os.path.join(result_directory, f'iter_{j}')
            os.makedirs(iter_directory, exist_ok=True)

            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                filename = os.path.join(iter_directory, f'results_plots_exp{fig_num-1}.png')
                fig.savefig(filename, bbox_inches='tight')
                plt.close(fig)  # Close the figure after saving to free up memory
            
            for i, result_dict in enumerate(results):
                mae_metrics = result_dict['MAE_Stream/eval_phase/test_stream']
                r2_vals = result_dict['R2_Stream/eval_phase/test_stream']
                all_mae.append(mae_metrics)
                all_r2.append(r2_vals)
                
            test_mae = min(all_mae)[0] if all_mae else float('inf')
            if test_mae <= best_mae:
                print(test_mae)
                print(best_mae)
                best_mae = test_mae
                io.cprint(f'Saving model iteration {j}')
                torch.save(fin_model, f'checkpoints/{args.exp_name}/iter_{j+1}/models/model_{args.strategy}.pt')

            io.cprint('MAE: {}'.format(all_mae))
            io.cprint('R2: {}'.format(all_r2))
            io.cprint('Class intro order: {}'.format(class_order))
            
            exp_name = args.exp_name
            os.makedirs(f'results/{exp_name}', exist_ok=True)

            # Save results to files
            with open(f'results/{exp_name}/mae.txt', 'a') as file:
                file.write(f"{all_mae}\n")

            with open(f'results/{exp_name}/r2.txt', 'a') as file:
                file.write(f"{all_r2}\n")
            
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            with open(f'results/{exp_name}/time.txt', 'a') as file:
                file.write(f"{elapsed_time:.2f}\n")

            # Print the timing metrics            
            io.cprint(f"Elapsed time for iteration {j+1}: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"An error occurred while running the script: {e}")
            print(f"Traceback:\n{tb}")

        print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
    # CL settings
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'mse', 'gmean'], help='training loss type')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='NAME',
                        help='Name of the experiment')
    parser.add_argument('--strategy', type=str, default='naive', metavar='STRATEGY',
                        choices=['naive', 'cumulative', 'replay', 'ewc', 'lwf','gem', 'agem', 'replay_ewc', 'gem_replay'],
                        help='Model to use, [naive, cumulative, replay, ewc, lwf, gem, agem]')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=91, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_iters', type=int, default=5,
                        help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='specify first iteration')
    parser.add_argument('--mem_size', type=int, default=100, metavar='HYPERPARAM',
                        help='Size of memory buffer used in Replay')
    parser.add_argument('--ewc_lambda', type=float, default=0.001, metavar='HYPERPARAM',
                        help='Lambda used for EWC')
    parser.add_argument('--gem_ppe', type=int, default=100, metavar='HYPERPARAM',
                        help='Patterns per experience GEM/ AGEM')
    parser.add_argument('--mem_strength', type=float, default=0.5, metavar='HYPERPARAM',
                        help='Strength of memory buffer GEM')
    parser.add_argument('--sample_size', type=int, default=10, metavar='HYPERPARAM',
                        help='Number of patterns in memory sample when computing reference gradient AGEM')
    parser.add_argument('--lwf_alpha', type=int, default=1, metavar='HYPERPARAM',
                        help='Distillation hyperparameter LwF')
    parser.add_argument('--lwf_temp', type=int, default=2, metavar='HYPERPARAM',
                        help='Softmax temperature for distillation LwF')    
    parser.add_argument('--model_path', type=str, default='', metavar='PATH',
                        help='Pretrained model path')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='BOOL',
                        help='True if using pretrained model')
    
    args = parser.parse_args()

    _init_(args)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')
        
    CLtrain(args, io)


# Example usage:
# python main.py --exp_name 'RAADL-PC_replay_pn20k_e200' --strategy 'replay' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8 
# python main.py --exp_name 'RAADL-PC_replay_pn20k_e200' --strategy 'gem' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8
# python main.py --exp_name 'RAADL-PC_debug' --strategy 'naive' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 12

# Pretrained model
# python main.py --exp_name 'RAADL-PC_PT_baseline_pn20k_e100' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=100 --num_iters 1 --batch_size 8 
# python main.py --exp_name 'RAADL-PC_PT_exptest' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=1 --num_iters 1 --batch_size 8
# python main.py --exp_name 'RAADL-PC_baseline_regpointnetPT-20k_epochs200_lr0-001-adapt' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=1 --num_iters 1 --batch_size 32
# Example usage
    # python main.py --exp_name 'RAADL-PC_replay_pn20k_e200' --strategy 'replay' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8 
    # python main.py --exp_name 'RAADL-PC_replay_pn20k_e200' --strategy 'gem' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 8
    # python main.py --exp_name 'RAADL-PC_debug' --strategy 'naive' --mem_size=1000 --epochs=200 --num_iters 4 --batch_size 12
    
    # Pretrained model
    # python main.py --exp_name 'RAADL-PC_PT_baseline_pn20k_e100' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=100 --num_iters 1 --batch_size 8 
    # python main.py --exp_name 'RAADL-PC_PT_exptest' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=1 --num_iters 1 --batch_size 8
    # python main.py --exp_name 'RAADL-PC_baseline_regpointnetPT-20k_epochs200_lr0-001-adapt' --strategy 'naive' --pretrained=True --model_path='models/PN_best.pth' --epochs=1 --num_iters 1 --batch_size 32

