# FNO Workflow Guide

This directory contains the essential files for the complete FNO (Fourier Neural Operator) workflow for Navier-Stokes equation prediction.

## 📁 Core Files

### Data Generation
- `generate_ns_data.py` - Generate Navier-Stokes 2D data
- `ns_data_v1e-5_N100.mat` - Generated dataset (100 samples, 64x64, 200 timesteps)

### Training Scripts
- `train_fno_notebook_replication.py` - Main FNO training script
- `train_fno_complete.py` - Complete training with data generation
- `train_lightning.py` - PyTorch Lightning version

### Analysis & Visualization
- `show_results.py` - Comprehensive results visualization
- `show_results_timestep.py` - Per-time-step analysis
- `replicate_spectral_bias.py` - Spectral bias replication
- `per_sample_spectral_analysis.py` - Individual sample analysis
- `visualize_generated_data.py` - Data visualization

### Model & Outputs
- `fno_model_notebook_replication.pth` - Trained FNO model
- `plots/` - All generated visualizations

## 🚀 Quick Start

### 1. Generate Data (if needed)
```bash
python generate_ns_data.py
```

### 2. Train FNO Model
```bash
python train_fno_notebook_replication.py
```

### 3. Generate Results
```bash
python show_results.py
python replicate_spectral_bias.py
python per_sample_spectral_analysis.py
```

### 4. View Plots
All plots are saved in the `plots/` directory:
- `fno_comprehensive_results.png` - Main results
- `fno_spectral_bias_replication.png` - Spectral bias analysis
- `per_sample_analysis_*.png` - Individual sample analyses
- `timestep_*_analysis.png` - Time-step analyses

## 📊 Key Results

- **Overall relative error**: ~6.1%
- **Low-frequency error**: ~9.6%
- **High-frequency error**: ~58%
- **Spectral bias ratio**: 6.8x higher error at high frequencies

## 🗂️ Old Files

Non-essential files have been moved to `../old/` including:
- Original notebooks and documentation
- Experimental scripts
- Old visualization files
- Additional source code

This keeps the working directory focused on the core workflow.
