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

def analyze_per_sample_spectral_loss():
    """Generate per-sample visualizations with spectral analysis"""
    
    print("=== Per-Sample Spectral Analysis ===")
    
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
    
    # Generate per-sample analysis
    n_samples = len(target_data)
    print(f"Analyzing {n_samples} test samples...")
    
    for sample_idx in range(min(n_samples, 10)):  # Analyze first 10 samples
        print(f"Processing sample {sample_idx+1}/{min(n_samples, 10)}...")
        
        # Create figure for this sample
        fig = plt.figure(figsize=(20, 16))
        
        # Select middle time step for detailed analysis
        t_idx = 5  # Middle of the prediction horizon
        
        gt_field = target_data[sample_idx, :, :, t_idx]
        pred_field = prediction[sample_idx, :, :, t_idx]
        error_field = pred_field - gt_field
        
        # 1. Ground Truth field
        ax1 = plt.subplot(4, 4, 1)
        vmin_gt, vmax_gt = gt_field.min(), gt_field.max()
        im1 = ax1.imshow(gt_field, cmap='RdBu_r', origin='lower', vmin=vmin_gt, vmax=vmax_gt)
        ax1.set_title(f'GROUND TRUTH\nSample {sample_idx+1}, t={t_idx+11}', fontweight='bold', color='green')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Prediction field
        ax2 = plt.subplot(4, 4, 2)
        vmin_pred, vmax_pred = pred_field.min(), pred_field.max()
        im2 = ax2.imshow(pred_field, cmap='RdBu_r', origin='lower', vmin=vmin_pred, vmax=vmax_pred)
        ax2.set_title(f'PREDICTION\nSample {sample_idx+1}, t={t_idx+11}', fontweight='bold', color='blue')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Error field
        ax3 = plt.subplot(4, 4, 3)
        error_abs = np.abs(error_field)
        max_error = max(0.5, error_abs.max())
        im3 = ax3.imshow(error_abs, cmap='hot', origin='lower', vmin=0, vmax=max_error)
        ax3.set_title(f'ABSOLUTE ERROR\nSample {sample_idx+1}', fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Add error statistics
        mean_error = np.mean(error_abs)
        max_error_val = np.max(error_abs)
        ax3.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nMax: {max_error_val:.4f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Error FFT magnitude
        ax4 = plt.subplot(4, 4, 4)
        error_fft = np.fft.fft2(error_field)
        error_fft_mag = np.abs(np.fft.fftshift(error_fft))
        error_fft_log = np.log10(error_fft_mag + 1e-10)
        im4 = ax4.imshow(error_fft_log, cmap='hot', origin='lower')
        ax4.set_title(f'ERROR FFT LOG |F|\nSample {sample_idx+1}', fontweight='bold')
        ax4.set_xlabel('kx')
        ax4.set_ylabel('ky')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # Compute spectra for all time steps in this sample
        gt_spectra = []
        pred_spectra = []
        error_spectra = []
        
        for t in range(10):
            gt_t = target_data[sample_idx, :, :, t]
            pred_t = prediction[sample_idx, :, :, t]
            error_t = pred_t - gt_t
            
            _, spec_gt = compute_radial_energy_spectrum_paper_method(gt_t)
            _, spec_pred = compute_radial_energy_spectrum_paper_method(pred_t)
            _, spec_error = compute_radial_energy_spectrum_paper_method(error_t)
            
            gt_spectra.append(spec_gt)
            pred_spectra.append(spec_pred)
            error_spectra.append(spec_error)
        
        gt_spectra = np.array(gt_spectra)
        pred_spectra = np.array(pred_spectra)
        error_spectra = np.array(error_spectra)
        
        # 5. Radial spectrum comparison
        ax5 = plt.subplot(4, 4, 5)
        k = np.arange(32)
        
        mean_gt_spectrum = np.mean(gt_spectra, axis=0)
        mean_pred_spectrum = np.mean(pred_spectra, axis=0)
        
        valid_gt = mean_gt_spectrum > 1e-10
        valid_pred = mean_pred_spectrum > 1e-10
        
        ax5.loglog(k[valid_gt], mean_gt_spectrum[valid_gt], 'g-', label='Ground Truth', linewidth=2)
        ax5.loglog(k[valid_pred], mean_pred_spectrum[valid_pred], 'b--', label='FNO Prediction', linewidth=2)
        ax5.set_xlabel('Wavenumber')
        ax5.set_ylabel('Energy Spectrum')
        ax5.set_title('Radial Energy Spectrum', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error spectrum
        ax6 = plt.subplot(4, 4, 6)
        mean_error_spectrum = np.mean(error_spectra, axis=0)
        valid_error = mean_error_spectrum > 1e-10
        
        ax6.loglog(k[valid_error], mean_error_spectrum[valid_error], 'r-', linewidth=2)
        ax6.set_xlabel('Wavenumber')
        ax6.set_ylabel('Error Energy')
        ax6.set_title('Error Spectrum', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Relative spectral loss
        ax7 = plt.subplot(4, 4, 7)
        relative_error_spectrum = mean_error_spectrum / (mean_gt_spectrum + 1e-10)
        valid_rel = (relative_error_spectrum > 1e-10) & (relative_error_spectrum < 1e10)
        
        ax7.semilogx(k[valid_rel], relative_error_spectrum[valid_rel], 'purple', linewidth=2)
        ax7.set_xlabel('Wavenumber')
        ax7.set_ylabel('Relative Error')
        ax7.set_title('Relative Spectral Loss', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Add frequency regions
        ax7.axvspan(0, 8, alpha=0.2, color='blue', label='Low freq')
        ax7.axvspan(8, 24, alpha=0.2, color='orange', label='Mid freq')
        ax7.axvspan(24, 32, alpha=0.2, color='red', label='High freq')
        
        # 8. Time evolution of relative error
        ax8 = plt.subplot(4, 4, 8)
        time_steps = np.arange(11, 21)
        relative_errors_time = []
        
        for t in range(10):
            rel_error = np.linalg.norm(prediction[sample_idx, :, :, t] - target_data[sample_idx, :, :, t]) / \
                       np.linalg.norm(target_data[sample_idx, :, :, t])
            relative_errors_time.append(rel_error)
        
        ax8.plot(time_steps, relative_errors_time, 'ko-', linewidth=2, markersize=6)
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Relative L2 Error')
        ax8.set_title('Error Evolution', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 9-12: Show multiple time steps
        time_steps_show = [0, 3, 6, 9]
        for i, t_show in enumerate(time_steps_show):
            ax = plt.subplot(4, 4, 9 + i)
            
            # Show side-by-side comparison
            gt_show = target_data[sample_idx, :, :, t_show]
            pred_show = prediction[sample_idx, :, :, t_show]
            
            # Create combined visualization
            combined = np.concatenate([gt_show, pred_show], axis=1)
            vmin_combined = min(gt_show.min(), pred_show.min())
            vmax_combined = max(gt_show.max(), pred_show.max())
            
            im = ax.imshow(combined, cmap='RdBu_r', origin='lower', vmin=vmin_combined, vmax=vmax_combined)
            ax.axvline(x=63.5, color='black', linewidth=2)
            ax.set_title(f't={t_show+11}: GT | Pred', fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 13-16: Spectral analysis for different time steps
        for i, t_show in enumerate(time_steps_show):
            ax = plt.subplot(4, 4, 13 + i)
            
            # Compute spectra for this time step
            _, spec_gt_t = compute_radial_energy_spectrum_paper_method(target_data[sample_idx, :, :, t_show])
            _, spec_pred_t = compute_radial_energy_spectrum_paper_method(prediction[sample_idx, :, :, t_show])
            
            valid_gt_t = spec_gt_t > 1e-10
            valid_pred_t = spec_pred_t > 1e-10
            
            ax.loglog(k[valid_gt_t], spec_gt_t[valid_gt_t], 'g-', label='GT', linewidth=2)
            ax.loglog(k[valid_pred_t], spec_pred_t[valid_pred_t], 'b--', label='Pred', linewidth=2)
            ax.set_xlabel('Wavenumber')
            ax.set_ylabel('Energy')
            ax.set_title(f'Spectrum t={t_show+11}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add sample statistics
        overall_rel_error = np.mean(relative_errors_time)
        low_freq_rel_error = np.mean(relative_error_spectrum[1:9])
        high_freq_rel_error = np.mean(relative_error_spectrum[25:32])
        
        fig.suptitle(f'Per-Sample Analysis: Sample {sample_idx+1}\n' + 
                    f'Overall Error: {overall_rel_error:.4f} | ' +
                    f'Low-freq Error: {low_freq_rel_error:.3f} | ' +
                    f'High-freq Error: {high_freq_rel_error:.3f} | ' +
                    f'High/Low Ratio: {high_freq_rel_error/low_freq_rel_error:.1f}x',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual sample analysis
        filename = f'plots/per_sample_analysis_{sample_idx+1:02d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"  Saved: {filename}")
    
    print(f"\n=== Per-Sample Analysis Complete ===")
    print(f"Generated {min(n_samples, 10)} individual sample analyses in 'plots/' directory")
    
    # Summary statistics across all samples
    print(f"\n=== Summary Statistics ===")
    
    all_overall_errors = []
    all_low_freq_errors = []
    all_high_freq_errors = []
    
    for sample_idx in range(n_samples):
        # Compute overall error
        rel_errors = []
        for t in range(10):
            rel_error = np.linalg.norm(prediction[sample_idx, :, :, t] - target_data[sample_idx, :, :, t]) / \
                       np.linalg.norm(target_data[sample_idx, :, :, t])
            rel_errors.append(rel_error)
        all_overall_errors.append(np.mean(rel_errors))
        
        # Compute spectral errors
        gt_spectra = []
        pred_spectra = []
        error_spectra = []
        
        for t in range(10):
            gt_t = target_data[sample_idx, :, :, t]
            pred_t = prediction[sample_idx, :, :, t]
            error_t = pred_t - gt_t
            
            _, spec_gt = compute_radial_energy_spectrum_paper_method(gt_t)
            _, spec_pred = compute_radial_energy_spectrum_paper_method(pred_t)
            _, spec_error = compute_radial_energy_spectrum_paper_method(error_t)
            
            gt_spectra.append(spec_gt)
            pred_spectra.append(spec_pred)
            error_spectra.append(spec_error)
        
        gt_spectra = np.array(gt_spectra)
        pred_spectra = np.array(pred_spectra)
        error_spectra = np.array(error_spectra)
        
        mean_gt_spectrum = np.mean(gt_spectra, axis=0)
        mean_error_spectrum = np.mean(error_spectra, axis=0)
        
        relative_error_spectrum = mean_error_spectrum / (mean_gt_spectrum + 1e-10)
        
        low_freq_error = np.mean(relative_error_spectrum[1:9])
        high_freq_error = np.mean(relative_error_spectrum[25:32])
        
        all_low_freq_errors.append(low_freq_error)
        all_high_freq_errors.append(high_freq_error)
    
    all_overall_errors = np.array(all_overall_errors)
    all_low_freq_errors = np.array(all_low_freq_errors)
    all_high_freq_errors = np.array(all_high_freq_errors)
    
    print(f"Overall relative error: {np.mean(all_overall_errors):.4f} ± {np.std(all_overall_errors):.4f}")
    print(f"Low-frequency error: {np.mean(all_low_freq_errors):.4f} ± {np.std(all_low_freq_errors):.4f}")
    print(f"High-frequency error: {np.mean(all_high_freq_errors):.4f} ± {np.std(all_high_freq_errors):.4f}")
    print(f"High/Low frequency error ratio: {np.mean(all_high_freq_errors/all_low_freq_errors):.1f}x")
    
    best_sample = np.argmin(all_overall_errors)
    worst_sample = np.argmax(all_overall_errors)
    
    print(f"Best performing sample: {best_sample+1} (error: {all_overall_errors[best_sample]:.4f})")
    print(f"Worst performing sample: {worst_sample+1} (error: {all_overall_errors[worst_sample]:.4f})")

if __name__ == "__main__":
    analyze_per_sample_spectral_loss()
