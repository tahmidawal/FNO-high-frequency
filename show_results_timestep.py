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

def compute_radial_spectrum(field):
    """Compute radial power spectrum"""
    fft_field = np.fft.fft2(field)
    power = np.abs(fft_field)**2
    
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_radial = np.sqrt(kx_grid**2 + ky_grid**2)
    
    max_k = int(np.max(k_radial))
    radial_power = np.zeros(max_k + 1)
    
    for k in range(max_k + 1):
        mask = (k_radial >= k - 0.5) & (k_radial < k + 0.5)
        if np.sum(mask) > 0:
            radial_power[k] = np.mean(power[mask])
    
    return np.arange(max_k + 1), radial_power

def show_results_timestep():
    """Generate time-step-by-time-step visualization"""
    
    print("=== FNO Model Time Step Results Visualization ===")
    
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
    
    print("Loading trained model...")
    model = FNO2DTime().to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # Prepare test data
    train_size = 80
    eval_data = u_data[train_size:, :, :, :]
    
    # Select sample for analysis
    sample_idx = 0
    input_data = eval_data[sample_idx, :, :, :10].cpu().numpy()
    target_data = eval_data[sample_idx, :, :, 10:20].cpu().numpy()
    
    print("Generating model predictions...")
    with torch.no_grad():
        input_tensor = eval_data[sample_idx:sample_idx+1, :, :, :10]
        prediction = model(input_tensor).cpu().numpy()
    
    # Generate individual plots for each time step
    print("Generating time step plots...")
    
    for t_idx in range(10):
        print(f"Processing time step {t_idx+11}...")
        
        # Create figure for this time step
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Input field (corresponding input time step)
        ax1 = plt.subplot(2, 3, 1)
        input_field = input_data[:, :, min(t_idx, 9)]  # Use last input if t_idx > 9
        vmin_in, vmax_in = input_field.min(), input_field.max()
        im1 = ax1.imshow(input_field, cmap='RdBu_r', origin='lower', vmin=vmin_in, vmax=vmax_in)
        ax1.set_title(f'Input t={min(t_idx, 9)+1}', fontweight='bold', color='blue')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Ground Truth field
        ax2 = plt.subplot(2, 3, 2)
        gt_field = target_data[:, :, t_idx]
        vmin_gt, vmax_gt = gt_field.min(), gt_field.max()
        im2 = ax2.imshow(gt_field, cmap='RdBu_r', origin='lower', vmin=vmin_gt, vmax=vmax_gt)
        ax2.set_title(f'GROUND TRUTH t={t_idx+11}', fontweight='bold', color='green')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Prediction field
        ax3 = plt.subplot(2, 3, 3)
        pred_field = prediction[0, :, :, t_idx]
        vmin_pred, vmax_pred = pred_field.min(), pred_field.max()
        im3 = ax3.imshow(pred_field, cmap='RdBu_r', origin='lower', vmin=vmin_pred, vmax=vmax_pred)
        ax3.set_title(f'Prediction t={t_idx+11}', fontweight='bold', color='red')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 4. Error field
        ax4 = plt.subplot(2, 3, 4)
        error_field = np.abs(pred_field - gt_field)
        max_error = max(0.5, error_field.max())  # Ensure we have a reasonable scale
        im4 = ax4.imshow(error_field, cmap='hot', origin='lower', vmin=0, vmax=max_error)
        ax4.set_title(f'Absolute Error t={t_idx+11}', fontweight='bold')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Add error statistics
        mean_error = np.mean(error_field)
        max_error_val = np.max(error_field)
        ax4.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nMax: {max_error_val:.4f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Error FFT magnitude
        ax5 = plt.subplot(2, 3, 5)
        error_fft = np.fft.fft2(error_field)
        error_fft_mag = np.abs(np.fft.fftshift(error_fft))
        error_fft_log = np.log10(error_fft_mag + 1e-10)
        im5 = ax5.imshow(error_fft_log, cmap='hot', origin='lower')
        ax5.set_title(f'Error FFT Log |F|', fontweight='bold')
        ax5.set_xlabel('kx')
        ax5.set_ylabel('ky')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 6. Error Power Spectrum
        ax6 = plt.subplot(2, 3, 6)
        k_error, power_error = compute_radial_spectrum(error_field)
        valid_error = power_error > 1e-10
        
        if np.any(valid_error):
            ax6.loglog(k_error[valid_error], power_error[valid_error], 'orange', linewidth=2)
            ax6.set_xlabel('Wavenumber k')
            ax6.set_ylabel('Error Power Spectrum')
            ax6.set_title(f'Error Spectrum t={t_idx+11}', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add spectral slope if possible
            if len(k_error[valid_error]) > 5:
                # Fit power law in mid-range
                mid_range = (k_error >= 2) & (k_error <= 15) & valid_error
                if np.sum(mid_range) > 3:
                    k_mid = k_error[mid_range]
                    power_mid = power_error[mid_range]
                    log_k = np.log10(k_mid)
                    log_power = np.log10(power_mid)
                    slope = np.polyfit(log_k, log_power, 1)[0]
                    ax6.text(0.05, 0.95, f'Slope: {slope:.2f}', 
                            transform=ax6.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add overall title for this time step
        fig.suptitle(f'FNO Model Analysis - Time Step {t_idx+11}', fontsize=16, fontweight='bold')
        
        # Add data range information
        fig.text(0.02, 0.02, f'GT Range: [{gt_field.min():.2f}, {gt_field.max():.2f}] | Pred Range: [{pred_field.min():.2f}, {pred_field.max():.2f}]', 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual time step plot
        filename = f'plots/timestep_{t_idx+11:02d}_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"  Saved: {filename}")
    
    print(f"\n=== Time Step Analysis Complete ===")
    print(f"Generated {10} individual time step plots in 'plots/' directory")
    
    # Generate summary statistics
    print(f"\n=== Summary Statistics ===")
    
    # Calculate overall error metrics
    all_errors = []
    for t_idx in range(10):
        error_field = np.abs(prediction[0, :, :, t_idx] - target_data[:, :, t_idx])
        all_errors.append(np.mean(error_field))
    
    print(f"Mean error across time steps: {np.mean(all_errors):.6f}")
    print(f"Error standard deviation: {np.std(all_errors):.6f}")
    print(f"Best time step: {np.argmin(all_errors)+11} (error: {min(all_errors):.6f})")
    print(f"Worst time step: {np.argmax(all_errors)+11} (error: {max(all_errors):.6f})")

if __name__ == "__main__":
    show_results_timestep()
