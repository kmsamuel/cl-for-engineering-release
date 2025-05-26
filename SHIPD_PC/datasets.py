#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHIPD Point Cloud Dataset for Continual Learning

This module implements a PyTorch Dataset for loading and transforming 3D ship hull models 
from the SHIPD dataset stored as point clouds. It includes functionality to subsample or 
pad vertices to a fixed number of points, data augmentation methods, and normalization.

Author: Kaira Samuel
Contact: kmsamuel@mit.edu
"""

import os
import logging
from typing import Callable, Optional, Tuple, List
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataAugmentation:
    """Class encapsulating various data augmentation techniques for point clouds."""
    
    @staticmethod
    def translate_pointcloud(pointcloud: torch.Tensor, 
                           translation_range: Tuple[float, float] = (2./3., 3./2.)) -> torch.Tensor:
        """
        Translates the point cloud by random factors within a given range.

        Args:
            pointcloud: Input point cloud as torch.Tensor of shape [N, 3]
            translation_range: Tuple specifying range for translation factors

        Returns:
            Translated point cloud as torch.Tensor
        """
        xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return torch.tensor(translated_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, 
                         sigma: float = 0.01, 
                         clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the point cloud.

        Args:
            pointcloud: Input point cloud as torch.Tensor
            sigma: Standard deviation of Gaussian noise
            clip: Maximum absolute value for noise

        Returns:
            Jittered point cloud as torch.Tensor
        """
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud.

        Args:
            pointcloud: Input point cloud as torch.Tensor
            drop_rate: Percentage of points to randomly drop

        Returns:
            Point cloud with points dropped as torch.Tensor
        """
        num_drop = int(drop_rate * pointcloud.size(0))
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        return pointcloud[keep_indices, :]


class SHIPDPointCloudDataset(Dataset):
    """
    PyTorch Dataset for SHIPD point cloud data with support for continual learning.
    
    This dataset loads 3D ship hull models as point clouds and provides wave resistance 
    coefficient (Cw) predictions. It includes data augmentation, normalization, and 
    class assignment for continual learning scenarios.
    """

    def __init__(self, 
                 root_dir: str, 
                 csv_file: str, 
                 num_points: int = 20000,
                 transform: Optional[Callable] = None,
                 add_norm_feature: bool = False,
                 normalize_labels: bool = True,
                 train_ids_path: Optional[str] = None,
                 apply_augmentation: bool = True):
        """
        Initialize the SHIPD Point Cloud Dataset.

        Args:
            root_dir: Directory containing point cloud files (.pt format)
            csv_file: Path to CSV file with metadata and labels
            num_points: Fixed number of points to sample from each model
            transform: Optional transform function to apply to samples
            add_norm_feature: Whether to add coordinate norm as additional feature
            normalize_labels: Whether to apply z-score normalization to labels
            train_ids_path: Path to file containing training IDs for normalization
            apply_augmentation: Whether to apply data augmentation during training
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.add_norm_feature = add_norm_feature
        self.apply_augmentation = apply_augmentation
        self.augmentation = DataAugmentation()
        
        # Load metadata
        try:
            self.data_frame = pd.read_csv(csv_file)
            logging.info(f"Loaded {len(self.data_frame)} samples from {csv_file}")
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        # Setup label normalization
        self.normalization_params = None
        if normalize_labels:
            self._setup_label_normalization(train_ids_path)
        
        # Assign class targets for continual learning
        self.targets = self._assign_class_targets()

    def _setup_label_normalization(self, train_ids_path: Optional[str]) -> None:
        """Setup label normalization parameters based on training data."""
        if train_ids_path and os.path.exists(train_ids_path):
            # Load training IDs
            with open(train_ids_path, 'r') as f:
                train_ids = [line.strip() for line in f.readlines()]
            
            # Filter to get only training data
            train_mask = self.data_frame['design_id'].isin(train_ids)
            train_labels = self.data_frame.loc[train_mask, 'Cw'].values
            
            # Compute normalization parameters on training data only
            self.normalization_params = {
                'mean': float(np.mean(train_labels)),
                'std': float(np.std(train_labels))
            }
            logging.info(f"Label normalization - Mean: {self.normalization_params['mean']:.4f}, "
                        f"Std: {self.normalization_params['std']:.4f}")
        else:
            # Fallback to using all data
            all_labels = self.data_frame['Cw'].values
            self.normalization_params = {
                'mean': float(np.mean(all_labels)),
                'std': float(np.std(all_labels))
            }
            logging.warning("Using all data for label normalization (not recommended)")

    def _assign_class_targets(self) -> DataAttribute:
        """
        Assign class targets based on Cw value quartiles for continual learning.
        
        Returns:
            DataAttribute containing class targets (0-3)
        """
        labels = self.data_frame['Cw'].to_numpy()
        design_ids = self.data_frame['design_id'].to_numpy()
        
        # Calculate quartiles
        q25 = np.quantile(labels, 0.25)
        q50 = np.quantile(labels, 0.50)
        q75 = np.quantile(labels, 0.75)
        
        logging.info(f'Quartile breakdown: [{labels.min():.4f}, {q25:.4f}, '
                    f'{q50:.4f}, {q75:.4f}, {labels.max():.4f}]')

        # Assign classes based on quartiles
        targets = []
        class_counts = [0, 0, 0, 0]
        
        for label in labels:
            if label <= q25:
                target = 0
            elif label <= q50:
                target = 1
            elif label <= q75:
                target = 2
            else:
                target = 3
            
            targets.append(target)
            class_counts[target] += 1

        logging.info(f"Samples per class: {class_counts}")
        
        targets_tensor = torch.tensor(targets, dtype=torch.int32)
        return DataAttribute(targets_tensor, name="targets")

    def _normalize_point_cloud(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply min-max normalization to point cloud coordinates.
        
        Args:
            data: Point cloud tensor of shape [N, 3]
            
        Returns:
            Normalized point cloud tensor
        """
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
        
        return (data - min_vals) / range_vals

    def _load_point_cloud(self, design_id: str) -> Optional[torch.Tensor]:
        """
        Load point cloud from file.
        
        Args:
            design_id: Unique identifier for the design
            
        Returns:
            Point cloud tensor or None if loading fails
        """
        load_path = os.path.join(self.root_dir, f"{design_id}.pt")
        
        if not os.path.exists(load_path) or os.path.getsize(load_path) == 0:
            return None
            
        try:
            return torch.load(load_path, map_location='cpu')
        except (EOFError, RuntimeError) as e:
            logging.warning(f"Failed to load point cloud {load_path}: {e}")
            return None

    def _sample_or_pad_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Subsample or pad vertices to fixed number of points.
        
        Args:
            vertices: Point cloud vertices tensor
            
        Returns:
            Standardized point cloud with self.num_points vertices
        """
        num_vertices = vertices.size(0)
        
        if num_vertices > self.num_points:
            # Subsample
            indices = np.random.choice(num_vertices, self.num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < self.num_points:
            # Pad with zeros
            padding = torch.zeros((self.num_points - num_vertices, 3), dtype=torch.float32)
            vertices = torch.cat((vertices, padding), dim=0)
            
        return vertices

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized predictions back to original scale.
        
        Args:
            predictions: Normalized prediction values
            
        Returns:
            Denormalized predictions
        """
        if self.normalization_params is None:
            return predictions
            
        return predictions * self.normalization_params['std'] + self.normalization_params['mean']

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (point_cloud, label, class_target)
            - point_cloud: Tensor of shape [3/4, num_points] or [num_points, 3/4]
            - label: Wave resistance coefficient (Cw) value
            - class_target: Class label for continual learning (0-3)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Handle corrupted or missing data by trying next samples
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                current_idx = (idx + attempt) % len(self.data_frame)
                row = self.data_frame.iloc[current_idx]
                design_id = row['design_id']
                
                # Load point cloud
                vertices = self._load_point_cloud(design_id)
                if vertices is None:
                    continue
                
                # Standardize number of points
                vertices = self._sample_or_pad_vertices(vertices)
                
                # Apply augmentations during training
                if self.apply_augmentation:
                    vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                    vertices = self.augmentation.jitter_pointcloud(vertices)
                
                # Apply optional transforms
                if self.transform:
                    vertices = self.transform(vertices)
                
                # Normalize point cloud coordinates
                vertices_normalized = self._normalize_point_cloud(vertices)
                
                # Get label and apply normalization if needed
                label_value = float(row['Cw'])
                if self.normalization_params is not None:
                    label_value = ((label_value - self.normalization_params['mean']) / 
                                 self.normalization_params['std'])
                
                label = torch.tensor(label_value, dtype=torch.float32).view(-1)
                
                # Get class target
                target = self.targets[current_idx]
                
                # Add norm feature if requested
                if self.add_norm_feature:
                    # Transpose to [3, num_points] for feature concatenation
                    vertices_transposed = vertices_normalized.transpose(0, 1)
                    norms = torch.norm(vertices_transposed, dim=0, keepdim=True)
                    point_cloud = torch.cat((vertices_transposed, norms), dim=0)  # [4, num_points]
                else:
                    # Return as [3, num_points] for consistency
                    point_cloud = vertices_normalized.transpose(0, 1)
                
                return point_cloud, label, target
                
            except Exception as e:
                logging.warning(f"Error loading sample {current_idx}: {e}")
                continue
        
        # If all attempts failed, raise an error
        raise RuntimeError(f"Failed to load any valid sample starting from index {idx}")


# Compatibility alias for existing code
SHIPD_PC = SHIPDPointCloudDataset


if __name__ == '__main__':
    # Example usage and testing
    dataset_path = 'data/pointclouds'
    train_ids_path = 'data/train_val_test_splits/train_ids.txt'
    wave_coeff = 'data/CW_all.csv'
    num_points = 20000
    
    # Create dataset instance
    dataset = SHIPDPointCloudDataset(
        root_dir=dataset_path,
        csv_file=wave_coeff,
        num_points=num_points,
        add_norm_feature=True,
        normalize_labels=True,
        train_ids_path=train_ids_path
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Normalization params: {dataset.normalization_params}")
    
    # Test sample loading
    try:
        point_cloud, label, target = dataset[0]
        print(f"Sample 0 - Point cloud shape: {point_cloud.shape}, "
              f"Label: {label.item():.4f}, Target: {target}")
    except Exception as e:
        print(f"Error loading sample: {e}")