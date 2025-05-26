#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: SHIPD_Benchmark.py

Unified benchmark creation for SHIPD ship hydrodynamics dataset.
"""

import numpy as np
import logging
import os
import torch
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.scenarios.supervised import bin_incremental_benchmark
from datasets import create_shipd_dataset


def create_shipd_benchmark(data_dir='./data/', num_exp=4):
    """
    Create SHIPD benchmark with specified configuration.
    
    Args:
        data_dir: Directory containing the numpy data files
        num_exp: Number of experiences (default: 4)
        
    Returns:
        Avalanche benchmark object
    """
    
    print('=====> Preparing data...')
    
    # Create datasets
    datasets = create_shipd_dataset(data_dir, num_exp)
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    val_dataset = datasets['val']
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    
    # Convert to AvalancheDatasets
    train_dataset = AvalancheDataset(
        datasets=train_dataset, 
        data_attributes=[DataAttribute(train_dataset.targets, name='targets')]
    )
    test_dataset = AvalancheDataset(
        datasets=test_dataset, 
        data_attributes=[DataAttribute(test_dataset.targets, name='targets')]
    )
    val_dataset = AvalancheDataset(
        datasets=val_dataset, 
        data_attributes=[DataAttribute(val_dataset.targets, name='targets')]
    )
    
    # Create benchmark with bin incremental scenario
    # SHIPD uses 4 quantile-based classes
    class_order = [0, 1, 2, 3]
    
    benchmark = bin_incremental_benchmark(
        {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}, 
        class_order=class_order, 
        num_experiences=num_exp
    )
    
    return benchmark


# Pre-configured benchmark instance for easy import
def get_shipd_benchmark():
    """Get standard SHIPD benchmark (4 classes, bin incremental)"""
    return create_shipd_benchmark()


# Main benchmark selection - can be imported directly
SHIPD_Benchmark = get_shipd_benchmark()


if __name__ == "__main__":
    """Test benchmark creation"""
    
    print("Testing SHIPD benchmark...")
    benchmark = get_shipd_benchmark()
    
    for exp in benchmark.train_stream:
        print(f"Experience {exp.current_experience}")
        print(f"\tClass labels: {exp.classes_in_this_experience}")
        print(f"\tSample: {exp.dataset[0]}")
        break  # Just test first experience
    
    # Test creating CSV file for visualization
    import csv
    
    exp0 = benchmark.test_stream[0]
    exp1 = benchmark.test_stream[1]
    exp2 = benchmark.test_stream[2]
    exp3 = benchmark.test_stream[3]
    
    print(f"Exp 0 classes: {exp0.classes_in_this_experience}")
    print(f"Exp 2 classes: {exp2.classes_in_this_experience}")
    
    # Create sample CSV for visualization (as in original)
    csv_file = "data/in_vectors_figs_test.csv"
    
    with open(csv_file, mode="w", newline="") as file:
        pass
    
    # Loop through experiments and samples
    for exp_idx, exp in enumerate([exp0, exp1, exp2, exp3]):
        for i in range(5):
            in_vec, label, targ = exp.dataset[i]
            row = in_vec.tolist()
            
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

    print(f"Test CSV file created: {csv_file}")