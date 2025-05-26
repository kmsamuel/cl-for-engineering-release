#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel, Mohamed Elrefaie
@Contact: kmsamuel@mit.edu, mohamed.elrefaie@mit.edu
@File: datasets.py

Unified parametric DrivAerNet++ dataset module supporting both input and bin incremental scenarios.
"""
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from avalanche.benchmarks.utils import DataAttribute
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval


class DRIVAERNET_PAR(Dataset):
    """
    Unified PyTorch Dataset class for parametric DrivAerNet++ data.
    Supports both bin and input incremental target assignment strategies.
    """
    def __init__(self, df_data, num_exp, scenario='bin', cluster_csv_path='data/cluster_design_ids.csv'):
        """
        Initialize the parametric DrivAerNet++ dataset.
        
        Args:
            df_data: Pandas DataFrame with parametric data
            num_exp: Number of experiences/classes
            scenario: 'bin' (quantile-based) or 'input' (prefix/cluster-based)
            cluster_csv_path: Path to cluster CSV for input-based targets
        """
        self.df_data = df_data
        self.num_exp = num_exp
        self.scenario = scenario
        self.cluster_csv_path = cluster_csv_path
        self.num_per_quantile = [0] * num_exp
        self.quantiles = None
        
        # Assign targets based on scenario
        self.targets = self.assign_targets()

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        index = index % len(self.df_data)
        row = self.df_data.iloc[index]
        label = np.asarray([row['Average Cd']]).astype('float32')
        in_feats = np.asarray([row.drop(['Experiment', 'Average Cd', 'Std Cd']).values]).astype('float32')
        target = torch.tensor(self.targets[index])
        label = torch.tensor(label, dtype=torch.float32)
        in_feats = torch.tensor(in_feats, dtype=torch.float32).squeeze()

        return in_feats, label, target
    
    def assign_targets(self):
        """
        Assign targets based on scenario type.
        """
        if self.scenario == 'bin':
            return self._assign_bin_targets()
        elif self.scenario == 'input':
            return self._assign_input_targets()
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Must be 'bin' or 'input'")
    
    def _assign_bin_targets(self):
        """Assign quantile-based targets for bin incremental scenario."""
        self.num_per_quantile = [0] * self.num_exp
        labels = self.df_data['Average Cd'].to_numpy()

        # Compute bin edges based on number of experiences (num_exp)
        quantiles = [np.quantile(labels, q) for q in np.linspace(0, 1, self.num_exp + 1)]
        print('Quantile Breakdown:', quantiles)

        self.target = [None] * len(labels)

        for ind, el in enumerate(labels):
            for i in range(self.num_exp):
                # Include right endpoint in the last bin
                if quantiles[i] <= el < quantiles[i + 1] or (i == self.num_exp - 1 and el == quantiles[-1]):
                    self.target[ind] = i
                    self.num_per_quantile[i] += 1
                    break

        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class (bin scenario):", self.num_per_quantile)

        return self.targets
    
    def _assign_input_targets(self):
        """Assign prefix-based targets for input incremental scenario."""
        self.num_per_quantile = [0] * 3  # Typically 3 classes for input scenario
        experiment_ids = self.df_data['Experiment'].to_numpy()
        
        self.target = [None] * len(experiment_ids)
        for ind, experiment_id in enumerate(experiment_ids):
            if 'E_S' in experiment_id:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
            elif 'F_S' in experiment_id:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
            elif 'N_S' in experiment_id:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
        
        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class (input scenario):", self.num_per_quantile)

        return self.targets
    
    def _assign_cluster_targets(self):
        self.num_per_quantile = [0] * 3
        design_ids = self.df_data['Experiment'].to_numpy()
        
        self.target = [None] * len(design_ids)
        for ind, design_id in enumerate(design_ids):
            if 'E_S' in design_id:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1

            elif 'F_S' in design_id:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1

            elif 'N_S' in design_id:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
     
        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class:", self.num_per_quantile)

        return self.targets


# Factory function for easy dataset creation
def create_parametric_dataset(scenario='bin', num_exp=4, csv_file="data/DrivAerNet_ParametricData.csv"):
    """
    Factory function to create appropriate parametric DrivAerNet++ dataset.
    
    Args:
        scenario: 'bin' or 'input'
        num_exp: Number of experiences (4 for bin, 3 for input typically)
        csv_file: Path to parametric data CSV
        
    Returns:
        DRIVAERNET_PAR dataset instance
    """
    df_data = pd.read_csv(csv_file)
    
    if scenario == 'input' and num_exp != 3:
        print(f"Warning: input scenario typically uses 3 experiences, but {num_exp} specified")
    
    return DRIVAERNET_PAR(
        df_data=df_data,
        num_exp=num_exp,
        scenario=scenario
    )


if __name__ == "__main__":
    # Test both scenarios
    print("Testing bin scenario...")
    bin_dataset = create_parametric_dataset('bin', 4)
    print(f"Dataset size: {len(bin_dataset)}")
    
    inputs, outputs, targets = bin_dataset[0]
    print(f"Input shape: {inputs.shape}")
    print(f"Output: {outputs}")
    print(f"Target: {targets}")
    
    print("\nTesting input scenario...")
    input_dataset = create_parametric_dataset('input', 3)
    print(f"Dataset size: {len(input_dataset)}")
    
    inputs, outputs, targets = input_dataset[0]
    print(f"Input shape: {inputs.shape}")
    print(f"Output: {outputs}")
    print(f"Target: {targets}")
    
    # Test data loading
    print("\nTesting data loading...")
    dataloader = DataLoader(bin_dataset, batch_size=2, shuffle=True)
    for i, sample in enumerate(dataloader):
        inputs, outputs, targets = sample
        print(f"Batch {i+1}:")
        print(f"  Inputs: {inputs.shape}")
        print(f"  Outputs: {outputs.shape}")
        print(f"  Targets: {targets.shape}")
        if i >= 2:  # Just test a few batches
            break