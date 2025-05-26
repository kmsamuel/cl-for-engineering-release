#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel
@Contact: kmsamuel@mit.edu
@File: datasets.py

Unified SHIPD dataset module for ship hydrodynamics parametric data.
"""
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from avalanche.benchmarks.utils import DataAttribute
import pandas as pd
import matplotlib.pyplot as plt


class SHIPD(Dataset):
    """Legacy SHIPD dataset class using CSV files."""
    def __init__(self, df_input, df_label, num_exp):
        self.df_input = df_input
        self.df_label = df_label
        self.targets = self.assign_targets()
        self.num_per_quantile = [0] * num_exp
        self.quantiles = None

    def __len__(self):
        return len(self.df_label)

    def __getitem__(self, index):
        index = index % len(self.df_label)
        row_lab = self.df_label.iloc[index]
        row_vec = self.df_input.iloc[index]
        label = np.asarray([row_lab['T/Dd=0.5 Fn=0.3']]).astype('float32')
        in_vec = np.asarray([row_vec.values]).astype('float32')
        target = self.targets[index]
        return in_vec, label, target
    
    def assign_targets(self):
        self.num_per_quantile = [0]*4
        labels = self.df_label['T/Dd=0.5 Fn=0.3'].to_numpy()
        lower_quantile = np.quantile(labels, 0.25)
        middle_quantile = np.quantile(labels, 0.50)
        upper_quantile = np.quantile(labels, 0.75)
        
        print('Quantile Breakdown:', [labels.min(), lower_quantile, middle_quantile, upper_quantile, labels.max()])

        self.target = [None] * len(labels)

        for ind, el in enumerate(labels):
            if el <= lower_quantile:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
            elif lower_quantile < el <= middle_quantile:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
            elif middle_quantile < el <= upper_quantile:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
            else:
                self.target[ind] = 3
                self.num_per_quantile[3] += 1
                
        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        
        return self.targets


class SHIPD_npy(Dataset):
    """
    Main SHIPD dataset class using numpy arrays for ship hydrodynamics data.
    Supports bin incremental learning with 4 quantile-based classes.
    """
    def __init__(self, data, labels, num_exp=4):
        """
        Initializes the SHIPD dataset.

        Args:
            data (array-like): The input data (ship design parameters).
            labels (array-like): The target values (wave resistance coefficients).
            num_exp (int): Number of experiences (default: 4).
        """
        self.data = data
        self.labels = labels
        self.num_exp = num_exp

        # Calculate quantiles for target assignment
        lower_quantile = np.quantile(self.labels, 0.25)
        median_quantile = np.quantile(self.labels, 0.50)
        upper_quantile = np.quantile(self.labels, 0.75)
        print('Quantile Breakdown:', [labels.min(), lower_quantile, median_quantile, upper_quantile, labels.max()])

        # Define quantiles including min and max
        self.quantiles = [self.labels.min(), lower_quantile, median_quantile, upper_quantile, self.labels.max()]
        self.target = [None] * len(self.labels)
        self.num_per_quantile = [0] * num_exp

        # Assign targets based on quantiles
        for ind, el in enumerate(self.labels):
            if el <= lower_quantile:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
            elif lower_quantile < el <= median_quantile:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
            elif median_quantile < el < upper_quantile:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
            else:
                self.target[ind] = 3
                self.num_per_quantile[3] += 1

        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class:", self.num_per_quantile)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (data_sample, label_sample, target_sample)
        """
        data_sample = self.data[index]
        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        target_sample = torch.tensor(self.target, dtype=torch.int32)[index]
        label_sample = self.labels[index]
        
        return data_sample, label_sample, target_sample


# Factory function for easy dataset creation
def create_shipd_dataset(data_dir='./data/', num_exp=4):
    """
    Factory function to create SHIPD dataset from numpy files.
    
    Args:
        data_dir: Directory containing the numpy data files
        num_exp: Number of experiences (default: 4)
        
    Returns:
        Dictionary with train, test, val datasets
    """
    # Load training data
    X_train = np.load(data_dir + 'X_num_train.npy')
    Y_train = np.load(data_dir + 'y_train.npy')[:, 0]
    Y_train = torch.tensor(Y_train).unsqueeze(1)
    train_dataset = SHIPD_npy(data=X_train, labels=Y_train, num_exp=num_exp)

    # Load test data
    X_test = np.load(data_dir + 'X_num_test.npy')
    Y_test = np.load(data_dir + 'y_test.npy')[:, 0]
    Y_test = torch.tensor(Y_test).unsqueeze(1)
    test_dataset = SHIPD_npy(data=X_test, labels=Y_test, num_exp=num_exp)

    # Load validation data
    X_val = np.load(data_dir + 'X_num_val.npy')
    Y_val = np.load(data_dir + 'y_val.npy')[:, 0]
    Y_val = torch.tensor(Y_val).unsqueeze(1)
    val_dataset = SHIPD_npy(data=X_val, labels=Y_val, num_exp=num_exp)
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    }


if __name__ == "__main__":
    # Test the unified dataset
    print("Testing SHIPD dataset creation...")
    datasets = create_shipd_dataset()
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    val_dataset = datasets['val']
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    
    # Test data loading
    data_sample, label_sample, target_sample = train_dataset[0]
    print(f"Data shape: {data_sample.shape}")
    print(f"Label: {label_sample}")
    print(f"Target: {target_sample}")
    
    # Test data loading with DataLoader
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for i, (inputs, labels, targets) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Inputs: {inputs.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Targets: {targets.shape}")
        if i >= 2:  # Just test a few batches
            break