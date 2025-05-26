# cl-for-engineering-release
Benchmarking engineering tasks in the continual learning setting.

# Point Cloud Continual Learning Framework

A comprehensive framework for continual learning experiments on 3D point cloud datasets, supporting multiple architectures and continual learning strategies for geometric deep learning tasks.

## Overview

This repository implements continual learning algorithms for point cloud-based regression tasks across three major datasets:
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
conda create -n cl_pointcloud python=3.8
conda activate cl_pointcloud
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

```
├── datasets.py                 # Clean dataset implementations
├── *_Benchmark.py              # Benchmark creation for each dataset
├── main.py                     # Main training scripts
├── pretrained_model*.py        # Pretrained model architectures
├── refinement_model.py         # Model refinement utilities
├── loss.py                     # Custom loss functions
├── run_all_strat_exps.sh      # Experiment runner scripts
├── models/                     # Pretrained model checkpoints
├── data/                       # Dataset directories
│   ├── shapenet_cars/
│   ├── pointclouds/           # SHIPD data
│   └── stl_pointclouds/       # RAADL data
├── checkpoints/               # Training checkpoints
├── results/                   # Experiment results
└── logs/                      # Training logs
```

## Quick Start

### 1. Prepare Your Data

Organize your datasets in the following structure:

```bash
data/
├── shapenet_cars/
│   ├── pointclouds/           # .pt files
│   └── train_val_test_splits/
├── pointclouds/               # SHIPD dataset
│   ├── *.pt files
│   └── CW_all.csv
└── stl_pointclouds/           # RAADL dataset
    ├── *.pt files
    └── all_data_2024_filtered.csv
```

### 2. Download Pretrained Models

Place pretrained models in the `models/` directory:
- `PN_best.pth` - Pretrained PointNet model
- `pointnext_best.pth` - Pretrained PointNeXt model (if available)

### 3. Run Basic Experiments

#### ShapeNet Cars
```bash
# Naive baseline
python main.py --exp_name "ShapeNet_naive_baseline" --strategy "naive" --epochs 150 --batch_size 8 --base_seed 42

# Experience Replay
python main.py --exp_name "ShapeNet_replay" --strategy "replay" --mem_size 1000 --epochs 150 --batch_size 8 --base_seed 42

# With pretrained model
python main.py --exp_name "ShapeNet_pretrained" --strategy "naive" --pretrained=True --model_path "models/PN_best.pth" --epochs 100 --batch_size 8
```

#### SHIPD (Ship Hull Design)
```bash
# Naive baseline with pretrained model
python main.py --exp_name "SHIPD_naive_pretrained" --strategy "naive" --pretrained "PointNet" --model_path "models/PN_best.pth" --epochs 150 --lr 0.0005 --batch_size 80

# EWC regularization
python main.py --exp_name "SHIPD_ewc" --strategy "ewc" --ewc_lambda 10000 --pretrained "PointNet" --model_path "models/PN_best.pth" --epochs 150 --batch_size 80

# Experience Replay
python main.py --exp_name "SHIPD_replay" --strategy "replay" --mem_size 1600 --pretrained "PointNet" --model_path "models/PN_best.pth" --epochs 150 --batch_size 32
```

#### RAADL (Aircraft Design)
```bash
# Naive baseline
python main.py --exp_name "RAADL_naive" --strategy "naive" --pretrained=True --model_path "models/PN_best.pth" --epochs 100 --batch_size 8

# A-GEM
python main.py --exp_name "RAADL_agem" --strategy "agem" --gem_ppe 100 --sample_size 10 --pretrained=True --model_path "models/PN_best.pth" --epochs 100
```

### 4. Run Comprehensive Experiments

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
| **Experience Replay** | Store and replay past samples | `--mem_size` |
| **EWC** | Elastic Weight Consolidation | `--ewc_lambda` |
| **A-GEM** | Averaged Gradient Episodic Memory | `--gem_ppe`, `--sample_size` |
| **LwF** | Learning without Forgetting | `--lwf_alpha`, `--lwf_temp` |

### Strategy-Specific Parameters

```bash
# Experience Replay
--strategy replay --mem_size 1600

# EWC 
--strategy ewc --ewc_lambda 10000

# A-GEM
--strategy agem --gem_ppe 100 --sample_size 10

# Learning without Forgetting
--strategy lwf --lwf_alpha 1.0 --lwf_temp 2.0
```

## Configuration Options

### Common Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--exp_name` | Experiment name | `exp` | Any string |
| `--strategy` | CL strategy | `naive` | `naive`, `cumulative`, `replay`, `ewc`, `agem`, `lwf` |
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

## Advanced Usage

### Custom Benchmarks

Create custom continual learning scenarios:

```python
from datasets import ShapeNetCars
from ShapeNetCars_Benchmark import create_shapenet_benchmark

# Custom class order
benchmark = create_shapenet_benchmark(
    num_experiences=4,
    class_order=[0, 1, 2, 3]  # Custom order
)
```

### Performance Tips

- Use SSD storage for faster data loading
- Enable `pin_memory=True` for GPU training
- Use `persistent_workers=True` for faster epoch transitions
- Monitor GPU memory usage with `nvidia-smi`

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{pointcloud_continual_learning,
  title={Point Cloud Continual Learning Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/your-repo}
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
