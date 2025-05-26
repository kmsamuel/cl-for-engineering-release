# cl-for-engineering-release
Benchmarking engineering tasks in the continual learning setting.

# Point Cloud Continual Learning Framework

A comprehensive framework for continual learning experiments on 3D point cloud datasets, supporting multiple arcSHIPDDhitectures and continual learning strategies for geometric deep learning tasks.

## Overview

This repository implements continual learning algorithms for point cloud-based regression tasks across five major datasets:
- **DrivAerNet**: Drag coefficient prediction for automotive design
- **DrivAerNet++**: Drag coefficient prediction for automotive design
- **ShapeNet Cars**: Drag coefficient prediction for automotive design
- **SHIPD**: Wave resistance coefficient prediction for ship hull design  
- **RAADL**: Aerodynamic coefficient prediction for aircraft design

## Features

- **Multiple Datasets**: Support for ShapeNet, SHIPD, and RAADL point cloud datasets
- **Continual Learning Strategies**: Implementation of state-of-the-art CL algorithms
- **Pretrained Models**: Support for pretrained PointNet architectures
- **Advanced LR Scheduling**: Configurable learning rate scheduling for optimal convergence
- **Comprehensive Evaluation**: Extensive metrics and visualization tools
- **Reproducible Experiments**: Seed management and experiment tracking

## Installation

### Prerequisites
```bash
# Python 3.8+
```

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
pip install trimesh pyvista


## Installing the Custom Avalanche Fork

To install the modified Avalanche library required for these benchmarks:

```bash
git clone https://github.com/kmsamuel/cl-for-engineering.git
cd cl-for-engineering/avalanche
pip install -e .

### Optional Dependencies

```bash
# For visualization
pip install open3d plotly

# For mesh processing
pip install pymeshlab
```

## Repository Structure

The repository is organized into three main benchmark directories (DRIVAERNET_Par, DRIVAERNET_PC, SHAPENET_PC/, SHIPD_Par/, SHIPD_PC/,and RAADL/), each containing a complete implementation for continual learning experiments on that specific dataset. Each benchmark directory includes its own main.py training script, datasets.py dataset implementation, benchmark creation file, and relevant model architectures. Shared resources include the models/ directory for pretrained checkpoints, data/ for organized datasets with train/validation/test splits, and auto-generated directories for checkpoints/, results/, and logs/. This modular structure allows researchers to work with individual benchmarks independently while maintaining consistency across implementations.

## Quick Start

### 1. Prepare Your Data

Each benchmark directory contains its own data/ folder that should be populated with the appropriate dataset files. For point cloud benchmarks (DrivAerNet, DrivAerNet++, ShapeNet, SHIPD, RAADL), place the .pt point cloud files and corresponding CSV files containing target values (drag coefficients, wave resistance, etc.) in this folder, along with train/test/validation split files. For the SHIPD_Par parametric dataset, the .npy files are provided for both inputs and outputs; for the DrivAerNet++ parametric dataset, a .csv file is provided as the input parameters. The datasets.py file in each benchmark automatically parses the data folder to create the dataset class, while the benchmark file handles the continual learning scenario creation and train/test/validation splitting.
### 2. Download Pretrained Models

Place pretrained models in the `models/` directory:
- `PN_best.pth` - Pretrained PointNet model
- `pointnext_best.pth` - Pretrained PointNeXt model (if available)

### 3. Run Experiments
Use the provided bash scripts to run multiple strategies:

```bash
# Run all strategies for a dataset
chmod +x run_all_strat_exps.sh
./run_all_strat_exps.sh
```

## Continual Learning Strategies

### Supported Algorithms

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **Naive** | Sequential training without mitigation | - |
| **Cumulative** | Joint training (upper bound) | - |
| **Experience Replay (ER)** | Store and replay past samples | `--mem_size` |
| **EWC** | Elastic Weight Consolidation | `--ewc_lambda` |
| **GEM** | Gradient Episodic Memory | `--gem_ppe` |

## Configuration Options

### Common Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--exp_name` | Experiment name | `exp` | Any string |
| `--strategy` | CL strategy | `naive` | `naive`, `cumulative`, `replay`, `ewc`, `gem`|
| `--epochs` | Epochs per experience | `100` | Integer |
| `--batch_size` | Training batch size | `32` | Integer |
| `--lr` | Learning rate | `0.001` | Float |
| `--loss` | Loss function | `mse` | `l1`, `mse`, `huber` |
| `--base_seed` | Random seed base | `42` | Integer |
| `--num_iters` | Experiment iterations | `5` | Integer |

### Model Options

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--pretrained` | Use pretrained model | `False` | `True`/`False` or `PointNet`/`PointNeXt` |
| `--model_path` | Pretrained model path | `""` | Path to `.pth` file |

### Dataset-Specific Options

```bash
# SHIPD specific
--normalize_labels    # Apply z-score normalization to labels
--use_validation     # Use validation set for hyperparameter tuning

# RAADL specific  
--use_validation     # Use validation set instead of training set
```

## Results and Analysis

### Output Structure

```
results/
└── experiment_name/
    ├── mae.txt              # MAE scores per iteration
    ├── r2.txt               # R² scores per iteration  
    ├── time.txt             # Training time per iteration
    └── figs/                # Generated plots
        └── iter_X/
            └── results_plots_exp*.png
```

### Metrics

- **MAE (Mean Absolute Error)**: Primary regression metric
- **R² Score**: Coefficient of determination
- **Training Time**: Wall-clock time per iteration
- **Forgetting**: Backward transfer measurement

### Visualization

The framework automatically generates:
- Learning curves per experience
- R² correlation plots  
- Training progression visualization
- Performance comparison across strategies

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{cl_for_engineering,
  title={Point Cloud Continual Learning Framework},
  author=Kaira Samuel,
  year={2025},
  url={https://github.com/kmsamuel/cl-for-engineering-release}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Avalanche](https://github.com/ContinualAI/avalanche) continual learning library
- [PointNet](https://github.com/charlesq34/pointnet) architecture
- Dataset contributors: DrivAerNet, ShapeNet, SHIPD, and RAADL datasets

## Contact

- **Author**: Kaira Samuel
- **Email**: kmsamuel@mit.edu
- **Institution**: MIT


The datasets used for benchmarking will be made availabled on the Harvard Dataverse.
