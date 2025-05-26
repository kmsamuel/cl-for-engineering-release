"""
ShapeNet Cars Continual Learning Benchmark

This module creates a continual learning benchmark for drag coefficient prediction
on ShapeNet car models. It splits the dataset into experiences based on drag 
coefficient quartiles for class-incremental learning.

Clean version that works with configurable continual learning setup.
"""

import numpy as np
import logging
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Subset, ConcatDataset, DataLoader

# Avalanche imports
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark, bin_incremental_benchmark
from avalanche.benchmarks import NCScenario

# Local imports
from datasets import ShapeNetCars

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_subset(dataset: ShapeNetCars, ids_file: str) -> Subset:
    """
    Create a subset of the dataset based on design IDs from a text file.
    
    Args:
        dataset: The full ShapeNetCars dataset
        ids_file: Path to text file containing design IDs (one per line)
        
    Returns:
        Subset of the dataset with proper targets for continual learning
    """
    try:
        with open(ids_file, 'r') as file:
            subset_ids = file.read().split()
        
        # Filter the dataset DataFrame based on subset IDs
        subset_indices = dataset.data_frame[dataset.data_frame['file'].isin(subset_ids)].index.tolist()
        sub_dataset = Subset(dataset, subset_indices)
        
        # Add targets for continual learning
        dataset.assign_targets()  # Ensure targets are assigned
        sub_targets = dataset.targets[subset_indices]
        sub_dataset.targets = torch.tensor(sub_targets, dtype=torch.int32)
        sub_dataset.targets = DataAttribute(sub_dataset.targets, name="targets")
        
        return sub_dataset
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")
    except Exception as e:
        logging.error(f"Error creating subset from {ids_file}: {e}")
        raise


def create_shapenet_cars_benchmark(
    dataset_path: str = 'data/cars',
    aero_coeff_file: str = 'data/drag_coeffs.csv',
    train_ids_file: str = 'data/train_test_splits/train_ids.txt',
    test_ids_file: str = 'data/train_test_splits/test_ids.txt', 
    val_ids_file: str = 'data/train_test_splits/val_ids.txt',
    num_points: int = 20000,
    add_norm_feature: bool = True,
    num_experiences: int = 4,
    class_order: list = None,
    use_validation_as_test: bool = True
) -> NCScenario:
    """
    Create a ShapeNet Cars continual learning benchmark.
    
    Args:
        dataset_path: Path to directory containing point cloud files
        aero_coeff_file: Path to CSV file with drag coefficients
        train_ids_file: Path to training split IDs
        test_ids_file: Path to test split IDs  
        val_ids_file: Path to validation split IDs
        num_points: Number of points per point cloud
        add_norm_feature: Whether to add norm feature (4th channel)
        num_experiences: Number of continual learning experiences
        class_order: Order of classes for experiences (default: [0,1,2,3])
        use_validation_as_test: Whether to use validation set as test set
        
    Returns:
        Avalanche NCScenario for continual learning
    """
    if class_order is None:
        class_order = [0, 1, 2, 3]
    
    print('=====> Creating ShapeNet Cars Benchmark...')
    
    # Create full dataset
    full_dataset = ShapeNetCars(
        root_dir=dataset_path, 
        csv_file=aero_coeff_file, 
        num_points=num_points, 
        pointcloud_exist=True, 
        output_dir='experience_ids', 
        add_norm_feature=add_norm_feature
    )
    
    # Create train/test/val subsets
    train_dataset = create_subset(full_dataset, train_ids_file)
    test_dataset = create_subset(full_dataset, test_ids_file)
    val_dataset = create_subset(full_dataset, val_ids_file)
    
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
    
    # Choose test set (validation or test)
    eval_dataset = val_dataset if use_validation_as_test else test_dataset
    
    # Create the continual learning benchmark
    benchmark = bin_incremental_benchmark({'train': train_dataset, 'test': val_dataset}, class_order=[0,1,2,3], num_experiences=4)
    
    print(f"Benchmark created with {num_experiences} experiences")
    print(f"Class order: {class_order}")
    print(f"Using {'validation' if use_validation_as_test else 'test'} set for evaluation")
    
    return benchmark


# Create the default benchmark instance
print('=====> Preparing ShapeNet Cars data...')

# Default configuration
DATASET_PATH = 'data/cars'
AERO_COEFF_FILE = 'data/drag_coeffs.csv'
NUM_POINTS = 20000

# Create the benchmark
ShapeNetCars_Benchmark = create_shapenet_cars_benchmark(
    dataset_path=DATASET_PATH,
    aero_coeff_file=AERO_COEFF_FILE,
    num_points=NUM_POINTS,
    add_norm_feature=True,  # Enable 4-channel input for pretrained models
    num_experiences=4,
    class_order=[0, 1, 2, 3],
    use_validation_as_test=True
)


# Testing and validation
if __name__ == "__main__":
    print("\n=====> Testing ShapeNet Cars Benchmark...")
    
    # Test the benchmark structure
    print(f"Number of training experiences: {len(ShapeNetCars_Benchmark.train_stream)}")
    print(f"Number of test experiences: {len(ShapeNetCars_Benchmark.test_stream)}")
    
    # Test first few samples from training set
    first_experience = ShapeNetCars_Benchmark.train_stream[0]
    first_dataset = first_experience.dataset
    
    print(f"\nFirst experience info:")
    print(f"  Experience ID: {first_experience.current_experience}")
    print(f"  Classes in this experience: {first_experience.classes_in_this_experience}")
    print(f"  Dataset size: {len(first_dataset)}")
    
    # Test data loading
    print(f"\nTesting data loading from first experience...")
    for i in range(min(3, len(first_dataset))):
        try:
            vertices, cd_value, target = first_dataset[i]
            print(f"  Sample {i}: vertices shape={vertices.shape}, cd={cd_value.item():.4f}, target={target}")
        except Exception as e:
            print(f"  Sample {i}: Error loading - {e}")
    
    print(f"\n ShapeNet Cars benchmark ready for continual learning experiments!")