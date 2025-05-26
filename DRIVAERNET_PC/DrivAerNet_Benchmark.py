#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel, Mohamed Elrefaie
@Contact: kmsamuel@mit.edu, mohamed.elrefaie@mit.edu
@File: DrivAerNet_Benchmark.py

Unified benchmark creation for DrivAerNet datasets supporting multiple variants and scenarios.
"""

import numpy as np
import logging
import os
import torch
from torch.utils.data import Subset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.scenarios.supervised import bin_incremental_benchmark
from datasets import create_drivaernet_dataset


def create_subset(dataset, ids_file):
    """Helper function to create subsets from IDs in text files"""
    try:
        with open(os.path.join(ids_file), 'r') as file:
            subset_ids = file.read().split()
        # Filter the dataset DataFrame based on subset IDs
        subset_indices = dataset.data_frame[dataset.data_frame['Design'].isin(subset_ids)].index.tolist()        
        sub_dataset = Subset(dataset, subset_indices)

        # Adding targets for CL
        sub_targets = dataset.targets[subset_indices]
        sub_dataset.targets = torch.tensor(sub_targets, dtype=torch.int32)
        sub_dataset.targets = DataAttribute(sub_dataset.targets, name="targets")
        
        return sub_dataset
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")


def create_drivaernet_benchmark(variant: str = 'drivaernet', 
                               target_type: str = 'bin', 
                               num_points: int = 20000, 
                               add_norm_feature: bool = False,
                               train_val_split: str = 'old'):
    """
    Create DrivAerNet benchmark with specified configuration.
    
    Args:
        variant: 'drivaernet' or 'drivaernet_plus'
        target_type: 'bin' or 'input' (only for drivaernet_plus)
        num_points: Number of points to sample
        add_norm_feature: Whether to add norm feature for pretrained models
        train_val_split: 'old' or 'new' - which train/val/test split to use
        
    Returns:
        Avalanche benchmark object
    """
    
    # Create the dataset
    full_dataset = create_drivaernet_dataset(
        variant=variant, 
        target_type=target_type, 
        num_points=num_points, 
        add_norm_feature=add_norm_feature
    )
    
    # Determine split paths
    if train_val_split == 'old':
        split_dir = 'data/train_val_test_splits_old'
    else:
        split_dir = 'data/train_val_test_splits'
    
    # Create subsets
    train_dataset = create_subset(full_dataset, f'{split_dir}/train_design_ids.txt')
    val_dataset = create_subset(full_dataset, f'{split_dir}/val_design_ids.txt')
    test_dataset = create_subset(full_dataset, f'{split_dir}/test_design_ids.txt')
    
    print('=====> Preparing data...')
    print(f"Training data size: {len(train_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    print(f"Val data size: {len(val_dataset)}")
    
    # Convert to AvalancheDatasets
    train_dataset = AvalancheDataset(
        datasets=train_dataset, 
        data_attributes=[DataAttribute(train_dataset.targets, name='targets')]
    )
    val_dataset = AvalancheDataset(
        datasets=val_dataset, 
        data_attributes=[DataAttribute(val_dataset.targets, name='targets')]
    )
    test_dataset = AvalancheDataset(
        datasets=test_dataset, 
        data_attributes=[DataAttribute(test_dataset.targets, name='targets')]
    )
    
    # Determine class order and number of experiences based on variant and target type
    if variant == 'drivaernet' and target_type == 'bin':
        class_order = [0, 1, 2, 3]
        num_experiences = 4
    elif variant == 'drivaernet_plus' and target_type == 'bin':
        class_order = [0, 1, 2, 3]
        num_experiences = 4
    elif variant == 'drivaernet_plus' and target_type == 'input':
        class_order = [0, 1, 2]
        num_experiences = 3
    else:
        raise ValueError(f"Invalid combination: variant={variant}, target_type={target_type}")
    
    # Create benchmark
    benchmark = bin_incremental_benchmark(
        {'train': train_dataset, 'test': test_dataset}, 
        class_order=class_order, 
        num_experiences=num_experiences
    )
    
    return benchmark


# Pre-configured benchmark instances for easy import
def get_drivaernet_benchmark():
    """Get standard DrivAerNet benchmark (4 classes, bin incremental)"""
    return create_drivaernet_benchmark(
        variant='drivaernet', 
        target_type='bin', 
        num_points=20000, 
        add_norm_feature=True,
        train_val_split='old'
    )


def get_drivaernet_plus_bin_benchmark():
    """Get DrivAerNet++ benchmark with bin incremental (4 classes)"""
    return create_drivaernet_benchmark(
        variant='drivaernet_plus', 
        target_type='bin', 
        num_points=20000, 
        add_norm_feature=True,
        train_val_split='new'
    )


def get_drivaernet_plus_class_benchmark():
    """Get DrivAerNet++ benchmark with class incremental (6 classes)"""
    return create_drivaernet_benchmark(
        variant='drivaernet_plus', 
        target_type='input', 
        num_points=20000, 
        add_norm_feature=True,
        train_val_split='new'
    )


# Main benchmark selection - can be imported directly
# Default to DrivAerNet standard for backward compatibility
DrivAerNet_Benchmark = get_drivaernet_benchmark()


if __name__ == "__main__":
    """Test benchmark creation"""
    
    print("Testing DrivAerNet benchmark...")
    benchmark = get_drivaernet_benchmark()
    
    for exp in benchmark.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tBin labels: {exp.classes_in_this_experience}")
        print(f"\tSample: {exp.dataset[0]}")
        break  # Just test first experience
    
    print("\nTesting DrivAerNet++ bin benchmark...")
    benchmark_plus_bin = get_drivaernet_plus_bin_benchmark()
    
    for exp in benchmark_plus_bin.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tBin labels: {exp.classes_in_this_experience}")
        break
        
    print("\nTesting DrivAerNet++ class benchmark...")
    benchmark_plus_class = get_drivaernet_plus_class_benchmark()
    
    for exp in benchmark_plus_class.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tClass labels: {exp.classes_in_this_experience}")
        break