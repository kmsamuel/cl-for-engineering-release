#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:54:56 2023

@author: Kaira Samuel
@contact: kmsamuel@mit.edu

This module defines a PyTorch Dataset for loading and transforming 3D car models from the ShapeNet Cars dataset.
It includes functionality for:
- Loading pre-processed point clouds (.pt files) or STL files
- Subsampling or padding vertices to a fixed number of points
- Data augmentation techniques (translation, jittering, point dropping)
- Quantile-based class assignment for continual learning
- Visualization methods for point clouds and meshes

This dataset is designed for continual learning experiments on aerodynamic drag coefficient prediction.
"""
import os
from io import BytesIO
import logging
import torch
import numpy as np
import pandas as pd
import trimesh
from torch.utils.data import Dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
import pyvista as pv
import seaborn as sns
from typing import Callable, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """
    @staticmethod
    def translate_pointcloud(pointcloud: torch.Tensor, translation_range: Tuple[float, float] = (2./3., 3./2.)) -> torch.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a torch.Tensor.
        """
        # Randomly choose translation factors and apply them to the pointcloud
        xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return torch.tensor(translated_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        # Add Gaussian noise and clip to the specified range
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        # Calculate the number of points to drop
        num_drop = int(drop_rate * pointcloud.size(0))
        # Generate random indices for points to drop
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        # Drop the points
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud


class ShapeNetCars(Dataset):
    """
    PyTorch Dataset class for ShapeNet Cars dataset, handling loading, transforming, and augmenting 3D car models.
    
    This dataset supports both STL file loading and pre-processed point cloud loading (.pt files).
    It provides configurable data augmentation and normalization options.
    """
    
    def __init__(self, root_dir: str, csv_file: str, num_points: int, transform: Optional[Callable] = None, 
                 pointcloud_exist: bool = True, output_dir: str = 'experience_ids', add_norm_feature: bool = False):
        """
        Initializes the ShapeNetCars dataset instance.

        Args:
            root_dir: Directory containing the point cloud files (.pt) or STL files for 3D car models.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
            pointcloud_exist: Whether point clouds (.pt files) exist or need to be loaded from STL.
            output_dir: Directory to save experience ID files for continual learning.
            add_norm_feature: Whether to add norm feature (4th channel) to point clouds.
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.pointcloud_exist = pointcloud_exist
        self.add_norm_feature = add_norm_feature
        self.cache = {}
        
        # Load metadata CSV file
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        # Initialize class-related attributes
        self.num_per_quantile = [0] * 4
        self.design_ids_per_class = [[] for _ in range(4)]
        self.augmentation = DataAugmentation()
        
        # Create output directory for experience IDs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Assign targets based on quantiles
        self.targets = self.assign_targets()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.

        Args:
            data: Input data as a torch.Tensor.

        Returns:
            Normalized data as a torch.Tensor.
        """
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized_data = (data - min_vals) / range_vals
        return normalized_data

    def z_score_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data using z-score normalization (standard score).
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        std_vals = data.std(dim=0, keepdim=True)
        
        # Avoid division by zero
        std_vals[std_vals == 0] = 1.0
        
        normalized_data = (data - mean_vals) / std_vals
        return normalized_data

    def mean_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [-1, 1] based on mean and range.
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        normalized_data = (data - mean_vals) / range_vals
        return normalized_data

    def _sample_or_pad_vertices(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.size(0)
        # Subsample the vertices if there are more than the desired number
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        # Pad with zeros if there are fewer vertices than desired
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 3), dtype=torch.float32)
            vertices = torch.cat((vertices, padding), dim=0)
        return vertices
    
    def assign_targets(self):
        """
        Assign class targets based on drag coefficient (Cd) quantiles for continual learning.
        
        Creates 4 classes based on quartiles of the drag coefficient distribution.
        """
        self.num_per_quantile = [0] * 4
        labels = self.data_frame['Cd'].to_numpy()
        design_ids = self.data_frame['file'].to_numpy()
        
        # Calculate quantiles for class assignment
        lower_quantile = np.quantile(labels, 0.25)
        middle_quantile = np.quantile(labels, 0.50)
        upper_quantile = np.quantile(labels, 0.75)
        
        print('Drag Coefficient Quantile Breakdown:', 
              [labels.min(), lower_quantile, middle_quantile, upper_quantile, labels.max()])

        self.target = [None] * len(labels)

        # Assign classes based on quantiles
        for ind, cd_value in enumerate(labels):
            design_id = design_ids[ind]
            
            if cd_value <= lower_quantile:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
                self.design_ids_per_class[0].append(design_id)
            elif lower_quantile < cd_value <= middle_quantile:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
                self.design_ids_per_class[1].append(design_id)
            elif middle_quantile < cd_value <= upper_quantile:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
                self.design_ids_per_class[2].append(design_id)
            else:
                self.target[ind] = 3
                self.num_per_quantile[3] += 1
                self.design_ids_per_class[3].append(design_id)

        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number of samples per class:", self.num_per_quantile)
        
        return self.targets
    
    def write_design_ids_to_files(self, output_dir: str):
        """
        Write each design_ids_per_class list to a separate text file.
        
        This is useful for creating reproducible continual learning splits.
        """
        for i, design_ids in enumerate(self.design_ids_per_class):
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"design_ids_class_{i}.txt")
            with open(file_path, 'w') as f:
                for design_id in design_ids:
                    f.write(f"{design_id}\n")
            print(f"Saved design IDs for class {i} to {file_path}")
        
    def _load_point_cloud(self, design_id: str) -> Optional[torch.Tensor]:
        """
        Load a pre-processed point cloud from a .pt file.
        
        Args:
            design_id: The design identifier for the point cloud file.
            
        Returns:
            Loaded point cloud tensor or None if loading fails.
        """
        load_path = os.path.join(self.root_dir, f"{design_id}.pt")
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                return torch.load(load_path)
            except (EOFError, RuntimeError) as e:
                logging.warning(f"Failed to load point cloud file {load_path}: {e}")
                return None
        else:
            logging.warning(f"Point cloud file {load_path} does not exist or is empty.")
            return None
        
    def __getitem__(self, idx: int, apply_augmentations: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset.

        Args:
            idx: Index of the sample to retrieve.
            apply_augmentations: Whether to apply data augmentations.

        Returns:
            Tuple containing:
            - Point cloud tensor (3 or 4 channels depending on add_norm_feature)
            - Drag coefficient value
            - Class target for continual learning
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        while True:
            row = self.data_frame.iloc[idx]
            design_id = row['file']
            cd_value = row['Cd']

            # Load point cloud data
            if self.pointcloud_exist:
                vertices = self._load_point_cloud(design_id)
                if vertices is None:
                    # Skip to next sample if loading fails
                    idx = (idx + 1) % len(self.data_frame)
                    continue
                vertices = self._sample_or_pad_vertices(vertices, self.num_points)
            else:
                # Load from STL file
                geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
                try:
                    mesh = trimesh.load(geometry_path, force='mesh')
                    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                    vertices = self._sample_or_pad_vertices(vertices, self.num_points)
                except Exception as e:
                    logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
                    raise

            # Apply data augmentations if enabled
            if apply_augmentations:
                vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                vertices = self.augmentation.jitter_pointcloud(vertices)

            # Apply optional transformations
            if self.transform:
                vertices = self.transform(vertices)

            # Normalize the features of the point cloud
            point_cloud_normalized = self.min_max_normalize(vertices)

            # Prepare output tensors
            cd_value = torch.tensor(float(cd_value), dtype=torch.float32).view(-1)
            target = self.targets[idx]
            
            # Add norm feature if requested (creates 4th channel)
            if self.add_norm_feature:
                norms = torch.norm(point_cloud_normalized.transpose(0, 1), dim=0, keepdim=True)
                pointclouds_with_norm = torch.cat((point_cloud_normalized.transpose(0, 1), norms), dim=0)
                return pointclouds_with_norm, cd_value, target

            # Return standard 3-channel point cloud
            return point_cloud_normalized.transpose(0, 1), cd_value, target

    # Visualization methods
    def visualize_point_cloud(self, idx: int):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx: Index of the design to visualize in the dataset.
        """
        # Retrieve vertices without augmentations for visualization
        vertices, cd_value, target = self.__getitem__(idx, apply_augmentations=False)
        vertices = vertices.transpose(0, 1).numpy()  # Convert back to [N, 3] format

        # Convert vertices to a PyVista PolyData object for visualization
        point_cloud = pv.PolyData(vertices[:, :3])  # Use only XYZ coordinates
        colors = vertices[:, 2]  # Use z-coordinate for color mapping
        point_cloud["colors"] = colors

        # Set up the PyVista plotter
        plotter = pv.Plotter()
        plotter.add_points(point_cloud, scalars="colors", cmap="Blues", point_size=3, render_points_as_spheres=True)
        plotter.enable_eye_dome_lighting()
        plotter.add_axes()
        
        # Set camera position for consistent viewing
        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                          (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                          (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]
        plotter.camera_position = camera_position
        
        plotter.add_text(f"Design: {self.data_frame.iloc[idx]['file']}\nCd: {cd_value.item():.4f}\nClass: {target}", 
                        position='upper_left')
        plotter.show()

    def visualize_augmentations(self, idx: int):
        """
        Visualizes various augmentations applied to the point cloud.

        Args:
            idx: Index of the sample in the dataset to be visualized.
        """
        # Get original point cloud
        vertices, _, _ = self.__getitem__(idx, apply_augmentations=False)
        original_vertices = vertices.transpose(0, 1).numpy()[:, :3]  # [N, 3]
        
        # Apply different augmentations
        translated_pc = self.augmentation.translate_pointcloud(original_vertices)
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        dropped_pc = self.augmentation.drop_points(jittered_pc)

        # Create 2x2 visualization
        plotter = pv.Plotter(shape=(2, 2))

        # Original
        plotter.subplot(0, 0)
        plotter.add_text("Original Point Cloud", font_size=10)
        plotter.add_mesh(pv.PolyData(original_vertices), color='black', point_size=3)

        # Translated
        plotter.subplot(0, 1)
        plotter.add_text("Translated Point Cloud", font_size=10)
        plotter.add_mesh(pv.PolyData(translated_pc.numpy()), color='lightblue', point_size=3)

        # Jittered
        plotter.subplot(1, 0)
        plotter.add_text("Jittered Point Cloud", font_size=10)
        plotter.add_mesh(pv.PolyData(jittered_pc.numpy()), color='lightgreen', point_size=3)

        # Dropped
        plotter.subplot(1, 1)
        plotter.add_text("Dropped Point Cloud", font_size=10)
        plotter.add_mesh(pv.PolyData(dropped_pc.numpy()), color='salmon', point_size=3)

        plotter.show()


# Example usage and testing
if __name__ == '__main__':
    # Example usage for ShapeNet Cars dataset
    dataset_path = 'data/cars'
    aero_coeff = 'data/drag_coeffs.csv'
    num_points = 20000
    
    # Create dataset instance
    shapenet_dataset = ShapeNetCars(
        root_dir=dataset_path, 
        csv_file=aero_coeff, 
        num_points=num_points, 
        pointcloud_exist=True, 
        output_dir='experience_ids', 
        add_norm_feature=True
    )

    # Write design IDs for reproducible splits
    shapenet_dataset.write_design_ids_to_files(output_dir='experience_ids')
    
    # Visualize data distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    df = pd.read_csv(aero_coeff)
    cd_values = df['Cd']

    plt.hist(cd_values, bins=30, color='blue', edgecolor='black', alpha=0.7, label='ShapeNet Cars')
    plt.title("Distribution of Drag Coefficients - ShapeNet Cars Dataset")
    plt.xlabel("Drag Coefficient (Cd)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(shapenet_dataset)}")
    print(f"Samples per class: {shapenet_dataset.num_per_quantile}")