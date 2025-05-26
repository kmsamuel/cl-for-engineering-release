#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: Kaira Samuel, Mohamed Elrefaie
@Contact: kmsamuel@mit.edu, mohamed.elrefaie@mit.edu
@File: datasets.py

Unified DrivAerNet dataset module supporting both DrivAerNet and DrivAerNet++ variants
with configurable target assignment strategies.
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
import matplotlib.pyplot as plt
from ast import literal_eval

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


class DrivAerNetDataset(Dataset):
    """
    Unified PyTorch Dataset class for DrivAerNet and DrivAerNet++ datasets.
    Supports both bin and class incremental target assignment strategies.
    """
    def __init__(self, root_dir: str, csv_file: str, num_points: int,
                 transform: Optional[Callable] = None, pointcloud_exist: bool = True,
                 output_dir='experience_ids', add_norm_feature: bool = False,
                 dataset_variant: str = 'drivaernet', target_type: str = 'bin',
                 cluster_csv_path: str = 'data/cluster_design_ids.csv'):
        """
        Initializes the DrivAerNetDataset instance.

        Args:
            root_dir: Directory containing the point cloud files.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
            pointcloud_exist: Whether point clouds are pre-computed or need STL loading.
            output_dir: Directory to save experience IDs.
            add_norm_feature: Whether to add norm feature for pretrained models.
            dataset_variant: 'drivaernet' or 'drivaernet_plus'
            target_type: 'bin' (quantile-based) or 'input' (prefix/cluster-based)
            cluster_csv_path: Path to cluster CSV for class-based targets.
        """
        self.root_dir = root_dir
        self.dataset_variant = dataset_variant
        self.target_type = target_type
        self.cluster_csv_path = cluster_csv_path
        
        # Attempt to load the metadata CSV file and log errors if unsuccessful
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        self.transform = transform
        self.num_points = num_points
        self.num_per_quantile = [0] * 4
        self.design_ids_per_class = [[] for _ in range(4)]
        self.augmentation = DataAugmentation()
        self.pointcloud_exist = pointcloud_exist
        self.add_norm_feature = add_norm_feature
        self.cache = {}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Assign targets based on variant and type
        self.targets = self.assign_targets()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.
        """
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data
    
    def z_score_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data using z-score normalization (standard score).
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        std_vals = data.std(dim=0, keepdim=True)
        normalized_data = (data - mean_vals) / std_vals
        return normalized_data

    def mean_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [-1, 1] based on mean and range.
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - mean_vals) / (max_vals - min_vals)
        return normalized_data

    def _sample_or_pad_vertices(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.
        """
        if vertices is None:
            print("Skipping sample due to None vertices.")
            return None
                
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
        Assign targets based on dataset variant and target type.
        """
        if self.dataset_variant == 'drivaernet' and self.target_type == 'bin':
            return self._assign_bin_targets()
        elif self.dataset_variant == 'drivaernet_plus' and self.target_type == 'bin':
            return self._assign_bin_targets()
        elif self.dataset_variant == 'drivaernet_plus' and self.target_type == 'input':
            return self._assign_cluster_targets()
        else:
            raise ValueError(f"Invalid combination: variant={self.dataset_variant}, type={self.target_type}")
    
    def _assign_bin_targets(self):
        self.num_per_quantile = [0]*4
        labels = self.data_frame['Average Cd'].to_numpy()
        design_ids = self.data_frame['Design'].to_numpy()
        
        lower_quantile = np.quantile(labels, 0.25)
        middle_quantile = np.quantile(labels, 0.50)
        upper_quantile = np.quantile(labels, 0.75)
        
        print('Quantile Breakdown:', [labels.min(), lower_quantile, middle_quantile, upper_quantile, labels.max()])

        self.target = [None] * len(labels)

        for ind, el in enumerate(labels):
            design_id = design_ids[ind]
            # if el <= lower_quantile:
            if el <= lower_quantile:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
                self.design_ids_per_class[0].append(design_id)  # Add design_id to class 0

            # elif lower_quantile < el <= upper_quantile:
            elif lower_quantile < el <= middle_quantile:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
                self.design_ids_per_class[1].append(design_id)  # Add design_id to class 1

                
            elif middle_quantile < el <= upper_quantile:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
                self.design_ids_per_class[2].append(design_id)  # Add design_id to class 2

            else:
                self.target[ind] = 3
                self.num_per_quantile[3] += 1
                self.design_ids_per_class[3].append(design_id)  # Add design_id to class 3

        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class:", self.num_per_quantile)

        return self.targets
    
    def _assign_cluster_targets(self):
        self.num_per_quantile = [0] * 3
        design_ids = self.data_frame['Design'].to_numpy()
        
        self.target = [None] * len(design_ids)
        for ind, design_id in enumerate(design_ids):
            if 'E_S' in design_id:
                self.target[ind] = 0
                self.num_per_quantile[0] += 1
                self.design_ids_per_class[0].append(design_id)  # Add design_id to class 0

            elif 'F_S' in design_id:
                self.target[ind] = 1
                self.num_per_quantile[1] += 1
                self.design_ids_per_class[1].append(design_id)  # Add design_id to class 0

            elif 'N_S' in design_id:
                self.target[ind] = 2
                self.num_per_quantile[2] += 1
                self.design_ids_per_class[2].append(design_id)  # Add design_id to class 0

                    
        self.targets = torch.tensor(self.target, dtype=torch.int32)
        self.targets = DataAttribute(self.targets, name="targets")
        print("Number per class:", self.num_per_quantile)

        return self.targets
    
    def write_design_ids_to_files(self, output_dir):
        """Write each design_ids_per_class list to a separate text file."""
        for i, design_ids in enumerate(self.design_ids_per_class):
            file_path = os.path.join(output_dir, f"design_ids_class_{i}.txt")
            with open(file_path, 'w') as f:
                for design_id in design_ids:
                    f.write(f"{design_id}\n")
            print(f"Saved design IDs for class {i} to {file_path}")

    def _load_point_cloud(self, design_id: str) -> Optional[torch.Tensor]:
        load_path = os.path.join(self.root_dir, f"{design_id}.pt")
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                return torch.load(load_path)
            except (EOFError, RuntimeError) as e:
                return None
        else:
            return None
        
    def __getitem__(self, idx: int, apply_augmentations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if idx in self.cache:
            return self.cache[idx]
        
        while True:
            row = self.data_frame.iloc[idx]
            design_id = row['Design']
            cd_value = row['Average Cd']
        
            if self.pointcloud_exist:
                vertices = self._load_point_cloud(design_id)
                vertices = self._sample_or_pad_vertices(vertices, self.num_points)

                if vertices is None:
                    idx = (idx + 1) % len(self.data_frame)
                    continue
            else:
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
            cd_value = torch.tensor(float(cd_value), dtype=torch.float32).view(-1)
            target = self.targets[idx]
            
            # Add Norm of Coordinates
            if self.add_norm_feature:
                norms = torch.norm(point_cloud_normalized.transpose(0,1), dim=0, keepdim=True)
                pointclouds_with_norm = torch.cat((point_cloud_normalized.transpose(0,1), norms), dim=0)
                return pointclouds_with_norm, cd_value, target

            return point_cloud_normalized.transpose(0,1), cd_value, target

    def visualize_point_cloud(self, idx, point_size=4, subsample_factor=0.5, cmap="Greys_r", 
                        save_path='figures/point_clouds', file_format='svg'):
        """
        Visualizes the point cloud for a specific design from the dataset with improved aesthetics.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        cd_value = row['Average Cd']

        vertices = self._load_point_cloud(design_id)
        vertices = self._sample_or_pad_vertices(vertices, self.num_points)
        vertices = vertices.numpy()
        
        point_cloud = pv.PolyData(vertices)
        z_values = vertices[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        normalized_z = (z_values - z_min) / (z_max - z_min) if z_max > z_min else z_values
        point_cloud["colors"] = normalized_z
        
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 1200])
        plotter.background_color = 'white'
        
        plotter.add_points(
            point_cloud, 
            scalars="colors", 
            cmap=cmap,
            point_size=point_size,
            render_points_as_spheres=True,
            opacity=0.85
        )
        
        plotter.enable_eye_dome_lighting()
        plotter.add_axes(interactive=False, line_width=2, color='gray')
        
        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)
        ]
        plotter.camera_position = camera_position
        
        os.makedirs(save_path, exist_ok=True)
        output_file = f'{save_path}/point_cloud_{design_id}_{idx}.{file_format}'
        plotter.save_graphic(output_file)
        
        print(f"Saved visualization to {output_file}")
        plotter.close()


# Factory function for easy dataset creation
def create_drivaernet_dataset(variant: str = 'drivaernet', target_type: str = 'bin', 
                             num_points: int = 20000, add_norm_feature: bool = False):
    """
    Factory function to create appropriate DrivAerNet dataset.
    
    Args:
        variant: 'drivaernet' or 'drivaernet_plus'
        target_type: 'bin' or 'input' (only for drivaernet_plus)
        num_points: Number of points to sample
        add_norm_feature: Whether to add norm feature
    """
    dataset_path = 'data/pointclouds'
    
    if variant == 'drivaernet':
        aero_coeff = 'data/aero_coeffs_4k.csv'
        if target_type != 'bin':
            raise ValueError("DrivAerNet only supports 'bin' target type")
    elif variant == 'drivaernet_plus':
        aero_coeff = 'data/aero_coeffs_plus.csv'
    else:
        raise ValueError("variant must be 'drivaernet' or 'drivaernet_plus'")
    
    return DrivAerNetDataset(
        root_dir=dataset_path, 
        csv_file=aero_coeff, 
        num_points=num_points, 
        pointcloud_exist=True, 
        add_norm_feature=add_norm_feature,
        dataset_variant=variant,
        target_type=target_type
    )


if __name__ == '__main__':
    # Test the unified dataset
    print("Testing DrivAerNet dataset...")
    dataset = create_drivaernet_dataset('drivaernet', 'bin', 20000, True)
    print(f"Dataset size: {len(dataset)}")
    
    point_cloud, label, target = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Label: {label}")
    print(f"Target: {target}")
    
    print("\nTesting DrivAerNet++ dataset with bin targets...")
    dataset_plus_bin = create_drivaernet_dataset('drivaernet_plus', 'bin', 20000, True)
    print(f"Dataset size: {len(dataset_plus_bin)}")
    
    print("\nTesting DrivAerNet++ dataset with class targets...")
    dataset_plus_class = create_drivaernet_dataset('drivaernet_plus', 'input', 20000, True)
    print(f"Dataset size: {len(dataset_plus_class)}")