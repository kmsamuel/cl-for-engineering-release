#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHIPD Point Cloud Continual Learning Benchmark

This module creates continual learning benchmarks from the SHIPD point cloud dataset.
It sets up train/validation/test splits and creates class-incremental scenarios
for evaluating continual learning algorithms on ship hull drag prediction.

Author: Kaira Samuel
Contact: kmsamuel@mit.edu
"""

import os
import logging
from typing import List, Optional
import torch
from torch.utils.data import Subset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from datasets import SHIPDPointCloudDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_dataset_subset(dataset: SHIPDPointCloudDataset, ids_file: str) -> Subset:
    """
    Create a dataset subset based on design IDs from a file.
    
    Args:
        dataset: The full SHIPD dataset
        ids_file: Path to file containing design IDs (one per line)
        
    Returns:
        Subset of the dataset containing only specified design IDs
        
    Raises:
        FileNotFoundError: If the IDs file doesn't exist
        ValueError: If no valid IDs are found
    """
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    
    try:
        with open(ids_file, 'r') as file:
            subset_ids = [line.strip() for line in file.readlines() if line.strip()]
        
        if not subset_ids:
            raise ValueError(f"No valid IDs found in {ids_file}")
        
        # Find indices in dataset that match the subset IDs
        subset_indices = dataset.data_frame[
            dataset.data_frame['design_id'].isin(subset_ids)
        ].index.tolist()
        
        if not subset_indices:
            raise ValueError(f"No matching samples found for IDs in {ids_file}")
        
        # Create subset
        subset = Subset(dataset, subset_indices)
        
        # Add targets for continual learning
        subset_targets = dataset.targets[subset_indices]
        subset.targets = DataAttribute(subset_targets, name="targets")
        
        logging.info(f"Created subset with {len(subset)} samples from {ids_file}")
        return subset
        
    except Exception as e:
        logging.error(f"Error creating subset from {ids_file}: {e}")
        raise


def create_shipd_benchmark(dataset_path: str = 'data/pointclouds',
                          csv_file: str = 'data/CW_all.csv',
                          train_ids_path: str = 'data/train_val_test_splits/train_ids.txt',
                          val_ids_path: str = 'data/train_val_test_splits/val_ids.txt',
                          test_ids_path: str = 'data/train_val_test_splits/test_ids.txt',
                          num_points: int = 20000,
                          num_experiences: int = 4,
                          class_order: Optional[List[int]] = None,
                          normalize_labels: bool = True,
                          add_norm_feature: bool = True,
                          use_validation: bool = False):
    """
    Create SHIPD continual learning benchmark.
    
    Args:
        dataset_path: Path to directory containing point cloud files
        csv_file: Path to CSV file with metadata and labels
        train_ids_path: Path to file with training design IDs
        val_ids_path: Path to file with validation design IDs
        test_ids_path: Path to file with test design IDs
        num_points: Number of points to sample from each point cloud
        num_experiences: Number of continual learning experiences
        class_order: Order of classes for incremental learning
        normalize_labels: Whether to normalize regression labels
        add_norm_feature: Whether to add coordinate norm as feature
        use_validation: Whether to use validation set instead of training set
        
    Returns:
        Avalanche benchmark scenario for continual learning
    """
    
    # Validate file paths
    required_files = [dataset_path, csv_file, train_ids_path, test_ids_path]
    if use_validation:
        required_files.append(val_ids_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Set default class order
    if class_order is None:
        class_order = [0, 1, 2, 3]
    
    # Create full dataset
    logging.info("Creating SHIPD dataset...")
    full_dataset = SHIPDPointCloudDataset(
        root_dir=dataset_path,
        csv_file=csv_file,
        num_points=num_points,
        add_norm_feature=add_norm_feature,
        normalize_labels=normalize_labels,
        train_ids_path=train_ids_path
    )
    
    # Create subsets
    logging.info("Creating dataset subsets...")
    
    if use_validation:
        # Use validation set for hyperparameter tuning
        train_subset = create_dataset_subset(full_dataset, val_ids_path)
        test_subset = create_dataset_subset(full_dataset, test_ids_path)
        logging.info("Using validation set for training (hyperparameter tuning mode)")
    else:
        # Normal train/test split
        train_subset = create_dataset_subset(full_dataset, train_ids_path)
        test_subset = create_dataset_subset(full_dataset, test_ids_path)
        logging.info("Using training set for training (normal mode)")
    
    # Create Avalanche datasets with normalization parameters
    train_avalanche = AvalancheDataset(
        datasets=train_subset,
        data_attributes=[train_subset.targets]
    )
    train_avalanche.normalization_params = full_dataset.normalization_params
    
    test_avalanche = AvalancheDataset(
        datasets=test_subset,
        data_attributes=[test_subset.targets]
    )
    test_avalanche.normalization_params = full_dataset.normalization_params
    
    # Create continual learning benchmark
    logging.info(f"Creating benchmark with {num_experiences} experiences, "
                f"class order: {class_order}")
    
    benchmark = class_incremental_benchmark(
        {'train': train_avalanche, 'test': test_avalanche},
        class_order=class_order,
        num_experiences=num_experiences
    )
    
    logging.info("SHIPD benchmark created successfully")
    return benchmark


# Create the default benchmark
def get_default_benchmark():
    """Get the default SHIPD benchmark configuration."""
    return create_shipd_benchmark(
        num_experiences=4,
        class_order=[0, 1, 2, 3],
        normalize_labels=False,  # Set to False to match original configuration
        add_norm_feature=True
    )


# For backward compatibility
SHIPD_PC_Benchmark = get_default_benchmark()


if __name__ == "__main__":
    # Test benchmark creation
    print("Creating SHIPD benchmark...")
    
    try:
        # Test default benchmark
        benchmark = get_default_benchmark()
        
        print(f"Benchmark created with {len(benchmark.train_stream)} training experiences")
        print(f"Test stream has {len(benchmark.test_stream)} experiences")
        
        # Test loading samples from first experience
        first_experience = benchmark.train_stream[0]
        print(f"First experience contains classes: {first_experience.classes_in_this_experience}")
        print(f"First experience has {len(first_experience.dataset)} samples")
        
        # Test sample loading
        sample_loader = torch.utils.data.DataLoader(
            first_experience.dataset, 
            batch_size=2, 
            shuffle=False
        )
        
        for batch_idx, (points, labels, targets) in enumerate(sample_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Points shape: {points.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Sample labels: {labels.tolist()}")
            print(f"  Sample targets: {targets.tolist()}")
            break
            
        print("Benchmark test completed successfully!")
        
    except Exception as e:
        print(f"Error testing benchmark: {e}")
        import traceback
        traceback.print_exc()