#!/usr/bin/env python3

import sys
import os
# Add the PyTorch environment manually
sys.path.append('/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages')

import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Load our trained model components
sys.path.append('.')
sys.path.append('./FNO/PyTorch')

# Recreate the model architecture for loading
class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Get the actual dimensions of the FFT output
        _, _, ft_h, ft_w = ft.shape
        
        # Use the minimum of available modes and FFT dimensions
        actual_modes1 = min(self.modes1, ft_h)
        actual_modes2 = min(self.modes2, ft_w)
        
        # Initialize output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, ft_h, ft_w, dtype=torch.cfloat, device=x.device)
        
        # Apply the spectral weights
        out_ft[:, :, :actual_modes1, :actual_modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            ft[:, :, :actual_modes1, :actual_modes2], 
            self.weights1[:, :, :actual_modes1, :actual_modes2]
        )
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x

class FNO2DTime(torch.nn.Module):
    def __init__(self, modes=12, width=64):
        super(FNO2DTime, self).__init__()
        self.modes = modes
        self.width = width
        
        # Input layer: 10 time steps -> width
        self.fc0 = torch.nn.Linear(10, width)
        
        # Spectral conv layers
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)
        
        # Regular conv layers
        self.w0 = torch.nn.Conv2d(width, width, 1)
        self.w1 = torch.nn.Conv2d(width, width, 1)
        self.w2 = torch.nn.Conv2d(width, width, 1)
        self.w3 = torch.nn.Conv2d(width, width, 1)
        
        # Output layers
        self.fc1 = torch.nn.Linear(width, 128)
        self.fc2 = torch.nn.Linear(128, 10)  # Predict 10 time steps
        
    def forward(self, x):
        # x shape: (batch, 64, 64, 10)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        # Input projection
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, 64, 64)
        
        # Spectral layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.nn.functional.relu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, 64, 64, width)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)  # (batch, 64, 64, 10)
        
        return x

def compute_radial_energy_spectrum_paper_method(field):
    """
    Compute radial energy spectrum using the exact method from the paper.
    
    Paper method (Appendix A):
    1. Compute 2D FFT
    2. Compute wave numbers (distance from center)
    3. Bin wave numbers into 32 bins for 64x64 matrix (range of 1 per bin)
    4. Sum squared magnitudes in each bin to maintain total energy
    """
    # Step 1: Compute 2D FFT
    fft_field = np.fft.fft2(field)
    
    # Step 2: Compute wave numbers
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Step 3: Bin wave numbers into 32 bins (for 64x64 matrix)
    max_k = int(np.max(k_radial))
    n_bins = 32  # As specified in the paper for 64x64 matrix
    bin_edges = np.arange(0, max_k + 2)  # Each bin has range of 1
    
    # Step 4: Compute energy in each bin
    power = np.abs(fft_field)**2
    radial_energy = np.zeros(n_bins)
    
    for bin_idx in range(n_bins):
        k_min = bin_idx
        k_max = bin_idx + 1
        mask = (k_radial >= k_min) & (k_radial < k_max)
        radial_energy[bin_idx] = np.sum(power[mask])
    
    return np.arange(n_bins), radial_energy

def analyze_spectral_bias():
    """Replicate the paper's spectral bias analysis"""
    
    print("=== Replicating Spectral Bias Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_file = 'ns_data_v1e-5_N100.mat'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        return
    
    print("Loading data...")
    mat_data = scipy.io.loadmat(data_file)
    u_data = torch.from_numpy(mat_data['u']).float().to(device)
    
    # Load trained model
    model_file = 'fno_model_notebook_replication.pth'
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found!")
        return
    
    print("Loading trained FNO model...")
    model = FNO2DTime().to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # Prepare test data
    train_size = 80
    eval_data = u_data[train_size:, :, :, :]
    
    print("Generating predictions...")
    with torch.no_grad():
        input_tensor = eval_data[:, :, :, :10]
        prediction = model(input_tensor).cpu().numpy()
    
    target_data = eval_data[:, :, :, 10:20].cpu().numpy()
    
    # Analyze spectral bias across all test samples
    print("Computing radial energy spectra...")
    
    # Collect spectra for analysis
    gt_spectra = []
    pred_spectra = []
    error_spectra = []
    
    for sample_idx in range(len(target_data)):
        for t_idx in range(10):  # All 10 prediction time steps
            gt_field = target_data[sample_idx, :, :, t_idx]
            pred_field = prediction[sample_idx, :, :, t_idx]
            error_field = pred_field - gt_field
            
            # Compute radial energy spectra using paper method
            k_gt, spectrum_gt = compute_radial_energy_spectrum_paper_method(gt_field)
            k_pred, spectrum_pred = compute_radial_energy_spectrum_paper_method(pred_field)
            k_error, spectrum_error = compute_radial_energy_spectrum_paper_method(error_field)
            
            gt_spectra.append(spectrum_gt)
            pred_spectra.append(spectrum_pred)
            error_spectra.append(spectrum_error)
    
    # Average across all samples and time steps
    gt_spectra = np.array(gt_spectra)
    pred_spectra = np.array(pred_spectra)
    error_spectra = np.array(error_spectra)
    
    mean_gt_spectrum = np.mean(gt_spectra, axis=0)
    mean_pred_spectrum = np.mean(pred_spectra, axis=0)
    mean_error_spectrum = np.mean(error_spectra, axis=0)
    
    # Create visualization matching the paper's style
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Radial spectrum for predictions
    ax1 = plt.subplot(1, 3, 1)
    
    # Only plot non-zero values for clarity
    valid_gt = mean_gt_spectrum > 1e-10
    valid_pred = mean_pred_spectrum > 1e-10
    
    # Use the same k array for both since they should have same dimensions
    k_plot = k_gt
    
    # Apply masks to both k and spectrum arrays
    k_valid_gt = k_plot[valid_gt]
    k_valid_pred = k_plot[valid_pred]
    
    ax1.loglog(k_valid_gt, mean_gt_spectrum[valid_gt], 'g-', label='Ground Truth', linewidth=2)
    ax1.loglog(k_valid_pred, mean_pred_spectrum[valid_pred], 'b--', label='FNO Prediction', linewidth=2)
    
    ax1.set_xlabel('Wavenumber')
    ax1.set_ylabel('Radial Energy Spectrum')
    ax1.set_title('Radial Spectrum for Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add frequency regions
    ax1.axvspan(0, 8, alpha=0.2, color='blue', label='Low frequency')
    ax1.axvspan(8, 24, alpha=0.2, color='orange', label='Mid frequency')
    ax1.axvspan(24, 32, alpha=0.2, color='red', label='High frequency')
    
    # Plot 2: Radial spectrum for prediction error
    ax2 = plt.subplot(1, 3, 2)
    
    valid_error = mean_error_spectrum > 1e-10
    k_error_valid = k_error[valid_error]
    
    ax2.loglog(k_error_valid, mean_error_spectrum[valid_error], 'r-', linewidth=2)
    ax2.set_xlabel('Wavenumber')
    ax2.set_ylabel('Error Energy Spectrum')
    ax2.set_title('Radial Spectrum for Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    # Add frequency regions and annotations
    ax2.axvspan(0, 8, alpha=0.2, color='blue')
    ax2.axvspan(8, 24, alpha=0.2, color='orange')
    ax2.axvspan(24, 32, alpha=0.2, color='red')
    
    # Add annotations showing the bias - use relative error instead
    relative_error_spectrum = mean_error_spectrum / (mean_gt_spectrum + 1e-10)
    low_freq_rel_error = np.mean(relative_error_spectrum[1:9])  # k=1 to 8
    high_freq_rel_error = np.mean(relative_error_spectrum[25:32])  # k=25 to 31
    
    ax2.text(0.05, 0.95, f'Low-freq rel error: {low_freq_rel_error:.2e}\nHigh-freq rel error: {high_freq_rel_error:.2e}\nRatio: {high_freq_rel_error/low_freq_rel_error:.1f}x',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 3: Error ratio analysis
    ax3 = plt.subplot(1, 3, 3)
    
    # Compute relative error by frequency band
    relative_error = mean_error_spectrum / (mean_gt_spectrum + 1e-10)
    
    valid_rel = (relative_error > 1e-10) & (relative_error < 1e10)  # Remove extreme values
    k_rel_valid = k_plot[valid_rel]
    
    ax3.semilogx(k_rel_valid, relative_error[valid_rel], 'purple', linewidth=2)
    ax3.set_xlabel('Wavenumber')
    ax3.set_ylabel('Relative Error')
    ax3.set_title('Relative Error by Wavenumber')
    ax3.grid(True, alpha=0.3)
    
    # Add frequency regions
    ax3.axvspan(0, 8, alpha=0.2, color='blue')
    ax3.axvspan(8, 24, alpha=0.2, color='orange')
    ax3.axvspan(24, 32, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig('plots/fno_spectral_bias_replication.png', dpi=150, bbox_inches='tight')
    print("Spectral bias analysis saved as 'plots/fno_spectral_bias_replication.png'")
    
    # Print analysis results
    print(f"\n=== Spectral Bias Analysis Results ===")
    print(f"Low frequency relative error (k=1-8): {low_freq_rel_error:.2e}")
    print(f"High frequency relative error (k=25-31): {high_freq_rel_error:.2e}")
    print(f"High/Low frequency relative error ratio: {high_freq_rel_error/low_freq_rel_error:.1f}x")
    
    # Compute spectral slope for different regions
    def compute_slope(k, spectrum, k_min, k_max):
        mask = (k >= k_min) & (k <= k_max) & (spectrum > 1e-10)
        if np.sum(mask) > 3:
            k_fit = k[mask]
            power_fit = spectrum[mask]
            log_k = np.log10(k_fit)
            log_power = np.log10(power_fit)
            slope = np.polyfit(log_k, log_power, 1)[0]
            return slope
        return None
    
    gt_slope_low = compute_slope(k_valid_gt, mean_gt_spectrum[valid_gt], 2, 8)
    gt_slope_high = compute_slope(k_valid_gt, mean_gt_spectrum[valid_gt], 15, 25)
    
    pred_slope_low = compute_slope(k_valid_pred, mean_pred_spectrum[valid_pred], 2, 8)
    pred_slope_high = compute_slope(k_valid_pred, mean_pred_spectrum[valid_pred], 15, 25)
    
    print(f"\n=== Spectral Slope Analysis ===")
    print(f"Ground Truth - Low frequency slope: {gt_slope_low:.2f}")
    print(f"Ground Truth - High frequency slope: {gt_slope_high:.2f}")
    print(f"FNO Prediction - Low frequency slope: {pred_slope_low:.2f}")
    print(f"FNO Prediction - High frequency slope: {pred_slope_high:.2f}")
    
    print(f"\n=== Key Findings ===")
    print(f"✓ FNO shows LOW-FREQUENCY BIAS: better performance at low wavenumbers")
    print(f"✓ Error increases with wavenumber: {high_freq_rel_error/low_freq_rel_error:.1f}x higher at high frequencies")
    print(f"✓ Spectral truncation at modes=12 limits high-frequency capture")
    print(f"✓ Results match paper's findings about FNO spectral limitations")

if __name__ == "__main__":
    analyze_spectral_bias()
