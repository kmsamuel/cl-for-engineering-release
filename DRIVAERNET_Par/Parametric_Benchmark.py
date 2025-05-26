#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: Parametric_Benchmark.py

Unified benchmark creation for parametric DrivAerNet++ supporting both input and bin incremental scenarios.
"""

import numpy as np
import logging
import os
import torch
from torch.utils.data import Subset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.scenarios.supervised import bin_incremental_benchmark
from datasets import create_parametric_dataset


def create_subset(dataset, ids_file):
    """Helper function to create subsets from IDs in text files"""
    try:
        with open(os.path.join(ids_file), 'r') as file:
            subset_ids = file.read().split()
        # Filter the dataset DataFrame based on subset IDs
        subset_indices = dataset.df_data[dataset.df_data['Experiment'].isin(subset_ids)].index.tolist()
        sub_dataset = Subset(dataset, subset_indices)

        # Adding targets for CL
        sub_targets = dataset.targets[subset_indices]
        sub_dataset.targets = torch.tensor(sub_targets, dtype=torch.int32)
        sub_dataset.targets = DataAttribute(sub_dataset.targets, name="targets")
        
        return sub_dataset
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")


def create_parametric_benchmark(scenario='bin', 
                               num_exp=None,
                               csv_file="data/DrivAerNet_ParametricData.csv",
                               train_ids_file='data/train_val_test_splits/train_ids.txt',
                               val_ids_file='data/train_val_test_splits/val_ids.txt',
                               test_ids_file='data/train_val_test_splits/test_ids.txt'):
    """
    Create parametric DrivAerNet++ benchmark with specified configuration.
    
    Args:
        scenario: 'bin' or 'input'
        num_exp: Number of experiences (auto-determined if None)
        csv_file: Path to parametric data CSV
        train_ids_file: Path to training IDs file
        val_ids_file: Path to validation IDs file  
        test_ids_file: Path to test IDs file
        
    Returns:
        Avalanche benchmark object
    """
    
    # Auto-determine number of experiences if not specified
    if num_exp is None:
        if scenario == 'bin':
            num_exp = 4
        elif scenario == 'input':
            num_exp = 3
        else:
            raise ValueError(f"Invalid scenario: {scenario}")
    
    # Create the dataset
    full_dataset = create_parametric_dataset(
        scenario=scenario,
        num_exp=num_exp,
        csv_file=csv_file
    )
    
    # Create subsets
    train_dataset = create_subset(full_dataset, train_ids_file)
    val_dataset = create_subset(full_dataset, val_ids_file)
    test_dataset = create_subset(full_dataset, test_ids_file)
    
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
    
    # Determine class order based on scenario
    if scenario == 'bin':
        class_order = list(range(num_exp))  # [0, 1, 2, 3] for 4 experiences
    elif scenario == 'input':
        class_order = [0, 1, 2]  # E_S, F_S, N_S
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
    
    # Create benchmark
    benchmark = bin_incremental_benchmark(
        {'train': train_dataset, 'test': test_dataset}, 
        class_order=class_order, 
        num_experiences=num_exp
    )
    
    return benchmark


# Pre-configured benchmark instances for easy import
def get_parametric_bin_benchmark():
    """Get parametric DrivAerNet++ benchmark with bin incremental (4 experiences)"""
    return create_parametric_benchmark(
        scenario='bin',
        num_exp=4
    )


def get_parametric_input_benchmark():
    """Get parametric DrivAerNet++ benchmark with input incremental (3 experiences)"""
    return create_parametric_benchmark(
        scenario='input',
        num_exp=3
    )


# Main benchmark selection - can be imported directly
# Default to bin incremental for backward compatibility
Parametric_Benchmark = get_parametric_bin_benchmark()


if __name__ == "__main__":
    """Test benchmark creation"""
    
    print("Testing parametric bin benchmark...")
    benchmark_bin = get_parametric_bin_benchmark()
    
    for exp in benchmark_bin.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tBin labels: {exp.classes_in_this_experience}")
        print(f"\tSample: {exp.dataset[0]}")
        break  # Just test first experience
    
    print("\nTesting parametric input benchmark...")
    benchmark_input = get_parametric_input_benchmark()
    
    for exp in benchmark_input.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tInput labels: {exp.classes_in_this_experience}")
        print(f"\tSample: {exp.dataset[0]}")
        break
        
    print("\nTesting custom benchmark...")
    custom_benchmark = create_parametric_benchmark(scenario='bin', num_exp=6)
    
    for exp in custom_benchmark.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tClass labels: {exp.classes_in_this_experience}")
        break