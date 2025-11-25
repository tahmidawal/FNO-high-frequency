# FNO High-Frequency Spectral Bias Analysis

This repository demonstrates the spectral bias limitations of Fourier Neural Operators (FNO) when solving the 2D Navier-Stokes equations. The analysis replicates findings from the paper ["Toward a Better Understanding of Fourier Neural Operators: Analysis and Improvement from a Spectral Perspective"](https://arxiv.org/abs/2404.07200), showing that FNO exhibits a distinct low-frequency bias with 6.8x higher relative errors at high frequencies.

## 🎯 Key Findings

- **Low-frequency relative error**: ~9.6%
- **High-frequency relative error**: ~58%  
- **Spectral bias ratio**: **6.8x** higher error at high frequencies
- **Overall prediction error**: ~6.1% relative L2 error

The analysis confirms that FNO's spectral truncation (12 modes) limits its ability to capture high-frequency turbulent structures, despite excellent performance at low frequencies.

## 📁 Repository Structure

```
├── generate_ns_data.py              # Navier-Stokes data generation
├── train_fno_notebook_replication.py # FNO model training
├── replicate_spectral_bias.py       # Spectral bias replication
├── per_sample_spectral_analysis.py  # Individual sample analysis
├── show_results.py                  # Comprehensive visualization
├── show_results_timestep.py         # Time-step analysis
├── visualize_generated_data.py      # Data visualization
├── FNO/                             # FNO model implementation
├── plots/                           # Generated visualizations
└── WORKFLOW.md                      # Detailed workflow guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch
- CUDA (recommended for training)
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/tahmidawal/FNO-high-frequency.git
cd FNO-high-frequency
pip install -r requirements.txt
```

### Complete Workflow

```bash
# 1. Generate Navier-Stokes data (64x64, 100 samples, 200 timesteps)
python generate_ns_data.py

# 2. Train the FNO model
python train_fno_notebook_replication.py

# 3. Replicate spectral bias analysis
python replicate_spectral_bias.py

# 4. Generate detailed per-sample analysis
python per_sample_spectral_analysis.py

# 5. Create comprehensive results visualization
python show_results.py

# 6. Generate time-step analysis
python show_results_timestep.py
```

## 📊 Generated Outputs

All visualizations are saved to the `plots/` directory:

- `fno_comprehensive_results.png` - Main results with adaptive scaling
- `fno_spectral_bias_replication.png` - Spectral bias analysis
- `per_sample_analysis_*.png` - Individual sample analyses (10 samples)
- `timestep_*_analysis.png` - Time-step analyses (steps 11-20)

## 🔬 Analysis Details

### Spectral Bias Replication

The `replicate_spectral_bias.py` script implements the exact methodology from the paper:

1. **Radial Energy Spectrum**: Uses 32 bins for 64×64 data
2. **Relative Error Analysis**: Computes `error_spectrum[k] / gt_spectrum[k]`
3. **Frequency Band Analysis**: Low (k=1-8), Mid (k=8-24), High (k=24-32)

### Per-Sample Analysis

The `per_sample_spectral_analysis.py` script generates 16-panel visualizations for each sample:

- **Spatial Analysis**: Ground truth, prediction, absolute error, error FFT
- **Spectral Analysis**: Energy spectra, error spectrum, relative spectral loss
- **Temporal Analysis**: Error evolution, multiple time steps
- **Frequency Analysis**: Spectral evolution across time

## 📈 Key Results

### Spectral Performance

| Frequency Range | Relative Error | Performance |
|-----------------|----------------|-------------|
| Low (k=1-8)     | ~9.6%         | Excellent   |
| High (k=25-31)  | ~58%          | Limited     |

### Model Architecture

- **Input**: 10 time steps of vorticity field
- **Output**: 10 predicted time steps  
- **Spectral Modes**: 12 (truncated at k=12)
- **Network**: 4 Fourier layers + MLP projections
- **Loss**: L2 relative error

## 🧪 Experimental Setup

- **Domain**: 2D periodic domain (0,1)²
- **Resolution**: 64×64 spatial grid
- **Viscosity**: ν = 1×10⁻⁵ (high Reynolds number)
- **Forcing**: f(x,y) = 0.1[sin(2π(x+y)) + cos(2π(x+y))]
- **Time**: T = 20, Δt = 1×10⁻⁴
- **Data**: 100 training samples, 20 test samples

## 🔧 Customization

### Modify Model Parameters

Edit `train_fno_notebook_replication.py`:

```python
model = FNO2DTime(modes=12, width=64)  # Adjust spectral modes and width
```

### Change Data Parameters

Edit `generate_ns_data.py`:

```python
resolution = 64      # Spatial resolution
N = 100             # Number of samples  
visc = 1e-5         # Viscosity coefficient
```

### Adjust Analysis

Edit `replicate_spectral_bias.py` to change frequency bands or visualization parameters.

## 📚 References

1. **Original Paper**: ["Toward a Better Understanding of Fourier Neural Operators: Analysis and Improvement from a Spectral Perspective"](https://arxiv.org/abs/2404.07200)
2. **FNO Original**: ["Fourier Neural Operator for Parametric Partial Differential Equations"](https://arxiv.org/abs/2010.08895)

## 🤝 Contributing

This repository focuses on reproducing the spectral bias analysis. Feel free to:
- Extend the analysis to other PDEs
- Implement SpecBoost ensemble method
- Add different neural operator architectures
- Improve visualization techniques

## 📄 License

This code is provided for research and educational purposes. Please cite the original papers if used in your research.

---

**Note**: Large binary files (`.mat`, `.pth`, `.png`) are excluded from git but can be generated by running the workflow scripts.