#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAADL Point Cloud Dataset for Continual Learning

This module implements a PyTorch Dataset for loading and transforming 3D airplane models 
from the RAADL dataset stored as point clouds. It includes functionality to subsample or 
pad vertices to a fixed number of points, data augmentation methods, and normalization.

Author: Kaira Samuel
Contact: kmsamuel@mit.edu

Adapted from the DrivAerNet code by Mohamed Elrefaie.
"""

import os
import logging
from typing import Callable, Optional, Tuple, Dict, Any
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


class RAADLPointCloudDataset(Dataset):
    """
    PyTorch Dataset for RAADL point cloud data with support for continual learning.
    
    This dataset loads 3D airplane models as point clouds and provides aerodynamic 
    coefficient predictions (CD, CL, CMy). It includes data augmentation, normalization, 
    and class assignment for continual learning scenarios.
    """

    def __init__(self, 
                 root_dir: str, 
                 csv_file: str, 
                 num_points: int = 20000,
                 transform: Optional[Callable] = None,
                 pointcloud_exist: bool = True,
                 output_dir: str = 'experience_ids',
                 add_norm_feature: bool = False,
                 apply_augmentation: bool = True):
        """
        Initialize the RAADL Point Cloud Dataset.

        Args:
            root_dir: Directory containing point cloud files (.pt format)
            csv_file: Path to CSV file with metadata and labels
            num_points: Fixed number of points to sample from each model
            transform: Optional transform function to apply to samples
            pointcloud_exist: Whether point cloud files exist
            output_dir: Directory for saving continual learning experience IDs
            add_norm_feature: Whether to add coordinate norm as additional feature
            apply_augmentation: Whether to apply data augmentation during training
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.pointcloud_exist = pointcloud_exist
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

        # Initialize continual learning attributes
        self.num_per_quantile = [0] * 4
        self.design_ids_per_class = [[] for _ in range(4)]
        self.cache = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Assign class targets for continual learning
        self.targets = self._assign_class_targets()

    def _assign_class_targets(self) -> DataAttribute:
        """
        Assign class targets based on CD value quartiles for continual learning.
        
        Returns:
            DataAttribute containing class targets (0-3)
        """
        labels = self.data_frame['CD'].to_numpy()
        design_ids = self.data_frame['Design ID'].to_numpy()
        
        # Calculate quartiles
        q25 = np.quantile(labels, 0.25)
        q50 = np.quantile(labels, 0.50)
        q75 = np.quantile(labels, 0.75)
        
        logging.info(f'CD quartile breakdown: [{labels.min():.4f}, {q25:.4f}, '
                    f'{q50:.4f}, {q75:.4f}, {labels.max():.4f}]')

        # Assign classes based on quartiles
        targets = []
        class_counts = [0, 0, 0, 0]
        
        for i, (label, design_id) in enumerate(zip(labels, design_ids)):
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
            self.design_ids_per_class[target].append(design_id)

        self.num_per_quantile = class_counts
        logging.info(f"Samples per class: {class_counts}")
        
        targets_tensor = torch.tensor(targets, dtype=torch.int32)
        return DataAttribute(targets_tensor, name="targets")

    def write_design_ids_to_files(self, output_dir: str):
        """Write design IDs for each class to separate text files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, design_ids in enumerate(self.design_ids_per_class):
            file_path = os.path.join(output_dir, f"design_ids_class_{i}.txt")
            with open(file_path, 'w') as f:
                for design_id in design_ids:
                    f.write(f"{design_id}\n")
            logging.info(f"Saved design IDs for class {i} to {file_path}")

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
        if vertices is None:
            logging.warning("Received None vertices, cannot process")
            return None
            
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

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_data, cd_label, class_target)
            - input_data: Dict with 'point_cloud' and 'flight_conditions'
            - cd_label: Drag coefficient (CD) value
            - class_target: Class label for continual learning (0-3)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Handle corrupted or missing data by trying next samples
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                current_idx = (idx + attempt) % len(self.data_frame)
                row = self.data_frame.iloc[current_idx]
                design_id = row['Design ID']
                
                # Get labels and flight conditions
                cd_value = float(row['CD'])
                flight_conditions = torch.tensor([
                    row['Alt_kft'], 
                    row['Re_L'], 
                    row['M_inf']
                ], dtype=torch.float32)
                
                # Load point cloud
                if self.pointcloud_exist:
                    vertices = self._load_point_cloud(design_id)
                    if vertices is None:
                        continue
                    vertices = self._sample_or_pad_vertices(vertices)
                    if vertices is None:
                        continue
                else:
                    # STL loading fallback (requires trimesh)
                    try:
                        import trimesh
                        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
                        mesh = trimesh.load(geometry_path, force='mesh')
                        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                        vertices = self._sample_or_pad_vertices(vertices)
                    except Exception as e:
                        logging.error(f"Failed to load STL file for {design_id}: {e}")
                        continue
                
                # Apply augmentations during training
                if self.apply_augmentation:
                    vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                    vertices = self.augmentation.jitter_pointcloud(vertices)
                
                # Apply optional transforms
                if self.transform:
                    vertices = self.transform(vertices)
                
                # Normalize point cloud coordinates
                vertices_normalized = self._normalize_point_cloud(vertices)
                
                # Prepare input data structure
                input_data = {
                    'point_cloud': vertices_normalized.transpose(0, 1),  # [3, num_points]
                    'flight_conditions': flight_conditions
                }
                
                # Add norm feature if requested
                if self.add_norm_feature:
                    # Compute norms and concatenate
                    norms = torch.norm(vertices_normalized.transpose(0, 1), dim=0, keepdim=True)
                    input_data['point_cloud'] = torch.cat((
                        vertices_normalized.transpose(0, 1), norms
                    ), dim=0)  # [4, num_points]
                
                # Get class target and create label tensor
                cd_label = torch.tensor(cd_value, dtype=torch.float32).view(-1)
                class_target = self.targets[current_idx]
                
                return input_data, cd_label, class_target
                
            except Exception as e:
                logging.warning(f"Error loading sample {current_idx}: {e}")
                continue
        
        # If all attempts failed, raise an error
        raise RuntimeError(f"Failed to load any valid sample starting from index {idx}")


# Compatibility alias for existing code
RAADLdataset = RAADLPointCloudDataset


if __name__ == '__main__':
    # Example usage and testing
    dataset_path = 'data/stl_pointclouds'
    aero_coeff = 'data/stl_pointclouds/all_data_2024_filtered.csv'
    num_points = 20000
    
    # Create dataset instance
    dataset = RAADLPointCloudDataset(
        root_dir=dataset_path,
        csv_file=aero_coeff,
        num_points=num_points,
        pointcloud_exist=True,
        add_norm_feature=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.num_per_quantile}")
    
    # Test sample loading
    try:
        input_data, cd_label, class_target = dataset[0]
        print(f"Sample 0:")
        print(f"  Point cloud shape: {input_data['point_cloud'].shape}")
        print(f"  Flight conditions shape: {input_data['flight_conditions'].shape}")
        print(f"  CD label: {cd_label.item():.4f}")
        print(f"  Class target: {class_target}")
    except Exception as e:
        print(f"Error loading sample: {e}")