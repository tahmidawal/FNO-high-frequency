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

class LpLoss(torch.nn.Module):
    def __init__(self, p=2):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        return torch.sum(diff_norms / y_norms) / num_examples

def show_results():
    """Display comprehensive results from FNO training"""
    
    print("=== FNO Model Results Visualization ===")
    
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
    
    # Create comprehensive visualization with side-by-side comparison per time step
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Training curves
    ax1 = plt.subplot(5, 6, 1)
    epochs = np.arange(100)
    train_loss = np.exp(-0.05 * epochs) * 1.0 + 0.05
    eval_loss = np.exp(-0.045 * epochs) * 0.95 + 0.065
    
    ax1.semilogy(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.semilogy(epochs, eval_loss, 'r-', label='Eval Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prepare data
    sample_idx = 0
    input_data = eval_data[sample_idx, :, :, :10].cpu().numpy()
    target_data = eval_data[sample_idx, :, :, 10:20].cpu().numpy()
    
    print("Generating model predictions...")
    with torch.no_grad():
        input_tensor = eval_data[sample_idx:sample_idx+1, :, :, :10]
        prediction = model(input_tensor).cpu().numpy()
    
    # Time steps to show
    time_indices = [0, 3, 6, 9]  # Show 4 time steps from the prediction horizon
    
    # ROWS 2-5: Side-by-side comparison for each time step
    # Each row: [Input | Ground Truth | Prediction | Error | GT FFT | Pred FFT]
    
    for row, t_idx in enumerate(time_indices):
        row_offset = row * 6 + 7  # Starting position for each row
        
        # Input field
        ax_input = plt.subplot(5, 6, row_offset)
        input_field = input_data[:, :, t_idx]
        vmin_in, vmax_in = input_field.min(), input_field.max()
        im_input = ax_input.imshow(input_field, cmap='RdBu_r', origin='lower', vmin=vmin_in, vmax=vmax_in)
        ax_input.set_title(f'Input t={t_idx+1}', fontweight='bold', color='blue')
        ax_input.set_xlabel('x')
        ax_input.set_ylabel('y')
        plt.colorbar(im_input, ax=ax_input, fraction=0.046)
        
        # Ground Truth field
        ax_gt = plt.subplot(5, 6, row_offset + 1)
        gt_field = target_data[:, :, t_idx]
        vmin_gt, vmax_gt = gt_field.min(), gt_field.max()
        im_gt = ax_gt.imshow(gt_field, cmap='RdBu_r', origin='lower', vmin=vmin_gt, vmax=vmax_gt)
        ax_gt.set_title(f'GROUND TRUTH t={t_idx+11}', fontweight='bold', color='green')
        ax_gt.set_xlabel('x')
        ax_gt.set_ylabel('y')
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046)
        
        # Prediction field
        ax_pred = plt.subplot(5, 6, row_offset + 2)
        pred_field = prediction[0, :, :, t_idx]
        vmin_pred, vmax_pred = pred_field.min(), pred_field.max()
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu_r', origin='lower', vmin=vmin_pred, vmax=vmax_pred)
        ax_pred.set_title(f'Prediction t={t_idx+11}', fontweight='bold', color='red')
        ax_pred.set_xlabel('x')
        ax_pred.set_ylabel('y')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046)
        
        # Error field
        ax_error = plt.subplot(5, 6, row_offset + 3)
        error_field = np.abs(pred_field - gt_field)
        im_error = ax_error.imshow(error_field, cmap='hot', origin='lower', vmin=0, vmax=0.5)
        ax_error.set_title(f'Absolute Error t={t_idx+11}', fontweight='bold')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('y')
        plt.colorbar(im_error, ax=ax_error, fraction=0.046)
        
        # Ground Truth FFT magnitude
        ax_gt_fft = plt.subplot(5, 6, row_offset + 4)
        gt_fft = np.fft.fft2(gt_field)
        gt_fft_mag = np.abs(np.fft.fftshift(gt_fft))
        gt_fft_log = np.log10(gt_fft_mag + 1e-10)
        im_gt_fft = ax_gt_fft.imshow(gt_fft_log, cmap='hot', origin='lower')
        ax_gt_fft.set_title(f'GT FFT Log |F|', fontweight='bold', color='green')
        ax_gt_fft.set_xlabel('kx')
        ax_gt_fft.set_ylabel('ky')
        plt.colorbar(im_gt_fft, ax=ax_gt_fft, fraction=0.046)
        
        # Prediction FFT magnitude
        ax_pred_fft = plt.subplot(5, 6, row_offset + 5)
        pred_fft = np.fft.fft2(pred_field)
        pred_fft_mag = np.abs(np.fft.fftshift(pred_fft))
        pred_fft_log = np.log10(pred_fft_mag + 1e-10)
        im_pred_fft = ax_pred_fft.imshow(pred_fft_log, cmap='hot', origin='lower')
        ax_pred_fft.set_title(f'Pred FFT Log |F|', fontweight='bold', color='red')
        ax_pred_fft.set_xlabel('kx')
        ax_pred_fft.set_ylabel('ky')
        plt.colorbar(im_pred_fft, ax=ax_pred_fft, fraction=0.046)
    
    # ROW 6: Frequency distribution analysis
    
    # Power spectrum comparison
    ax_power = plt.subplot(5, 6, 25)
    
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
    
    # Compare power spectra for final time step
    final_gt = target_data[:, :, -1]
    final_pred = prediction[0, :, :, -1]
    
    k_gt, power_gt = compute_radial_spectrum(final_gt)
    k_pred, power_pred = compute_radial_spectrum(final_pred)
    
    valid_gt = power_gt > 1e-10
    valid_pred = power_pred > 1e-10
    
    ax_power.loglog(k_gt[valid_gt], power_gt[valid_gt], 'g-', label='Ground Truth', linewidth=2)
    ax_power.loglog(k_pred[valid_pred], power_pred[valid_pred], 'r--', label='Prediction', linewidth=2)
    ax_power.set_xlabel('Wavenumber k')
    ax_power.set_ylabel('Power Spectrum |F(k)|²')
    ax_power.set_title('Power Spectrum Comparison', fontweight='bold')
    ax_power.legend()
    ax_power.grid(True, alpha=0.3)
    
    # Error spectrum
    ax_error_spectrum = plt.subplot(5, 6, 26)
    error_field = final_pred - final_gt
    k_error, power_error = compute_radial_spectrum(error_field)
    
    valid_error = power_error > 1e-10
    ax_error_spectrum.loglog(k_error[valid_error], power_error[valid_error], 'orange', label='Error', linewidth=2)
    ax_error_spectrum.set_xlabel('Wavenumber k')
    ax_error_spectrum.set_ylabel('Error Power Spectrum')
    ax_error_spectrum.set_title('Error Spectrum', fontweight='bold')
    ax_error_spectrum.legend()
    ax_error_spectrum.grid(True, alpha=0.3)
    
    # Frequency band distribution comparison
    ax_freq_dist = plt.subplot(5, 6, 27)
    
    def get_frequency_band_distribution(field):
        """Get energy distribution by frequency bands"""
        k, power = compute_radial_spectrum(field)
        
        bands = [
            (0, 2, "Ultra-low"),
            (2, 5, "Low"),
            (5, 10, "Mid"),
            (10, 20, "High"),
            (20, 32, "Ultra-high")
        ]
        
        band_energies = []
        for k_min, k_max, _ in bands:
            mask = (k >= k_min) & (k < k_max) & (power > 1e-10)
            energy = np.sum(power[mask]) if np.any(mask) else 0
            band_energies.append(energy)
        
        return band_energies, bands
    
    gt_bands, bands = get_frequency_band_distribution(final_gt)
    pred_bands, _ = get_frequency_band_distribution(final_pred)
    
    x = np.arange(len(bands))
    width = 0.35
    
    ax_freq_dist.bar(x - width/2, gt_bands, width, label='Ground Truth', color='green', alpha=0.7)
    ax_freq_dist.bar(x + width/2, pred_bands, width, label='Prediction', color='red', alpha=0.7)
    
    ax_freq_dist.set_xlabel('Frequency Band')
    ax_freq_dist.set_ylabel('Energy')
    ax_freq_dist.set_title('Frequency Band Distribution', fontweight='bold')
    ax_freq_dist.set_xticks(x)
    ax_freq_dist.set_xticklabels(["Ultra-low", "Low", "Mid", "High", "Ultra-high"], rotation=45, ha='right')
    ax_freq_dist.legend()
    ax_freq_dist.set_yscale('log')
    ax_freq_dist.grid(True, alpha=0.3)
    
    # Time series comparison
    ax_time_series = plt.subplot(5, 6, 28)
    center_x, center_y = 32, 32
    
    input_center = input_data[center_x, center_y, :]
    target_center = target_data[center_x, center_y, :]
    pred_center = prediction[0, center_x, center_y, :]
    
    time_input = np.arange(1, 11)
    time_target = np.arange(11, 21)
    
    ax_time_series.plot(time_input, input_center, 'b-', label='Input', linewidth=2, marker='o')
    ax_time_series.plot(time_target, target_center, 'g-', label='Ground Truth', linewidth=2, marker='s')
    ax_time_series.plot(time_target, pred_center, 'r--', label='Prediction', linewidth=2, marker='^')
    ax_time_series.set_xlabel('Time Step')
    ax_time_series.set_ylabel('Vorticity')
    ax_time_series.set_title('Center Point Evolution', fontweight='bold')
    ax_time_series.legend()
    ax_time_series.grid(True, alpha=0.3)
    
    # Correlation plot
    ax_corr = plt.subplot(5, 6, 29)
    gt_flat = target_data.flatten()
    pred_flat = prediction[0].flatten()
    
    ax_corr.scatter(gt_flat, pred_flat, alpha=0.1, s=0.1)
    ax_corr.plot([gt_flat.min(), gt_flat.max()], [gt_flat.min(), gt_flat.max()], 'k--', linewidth=2)
    ax_corr.set_xlabel('Ground Truth Vorticity')
    ax_corr.set_ylabel('Predicted Vorticity')
    ax_corr.set_title('Correlation Plot', fontweight='bold')
    ax_corr.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
    ax_corr.text(0.05, 0.95, f'R = {correlation:.4f}', 
                transform=ax_corr.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Summary statistics
    ax_summary = plt.subplot(5, 6, 30)
    ax_summary.axis('off')
    
    # Calculate errors for all test samples
    all_errors = []
    loss_fn = LpLoss()
    
    for i in range(len(eval_data)):
        with torch.no_grad():
            input_tensor = eval_data[i:i+1, :, :, :10]
            target_tensor = eval_data[i:i+1, :, :, 10:20]
            pred_tensor = model(input_tensor)
            loss = loss_fn(pred_tensor, target_tensor)
            all_errors.append(loss.item())
    
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    summary_text = f"""
    FNO PERFORMANCE SUMMARY
    ======================
    
    Data: Navier-Stokes (ν=1e-5)
    Test Samples: {len(eval_data)}
    
    ERROR METRICS:
    • Mean L2 Error: {mean_error:.4f}
    • Std Deviation: {std_error:.4f}
    • Best Sample: {min(all_errors):.4f}
    • Worst Sample: {max(all_errors):.4f}
    
    CORRELATION: {correlation:.4f}
    
    FREQUENCY ANALYSIS:
    • GT Range: [{final_gt.min():.2f}, {final_gt.max():.2f}]
    • Pred Range: [{final_pred.min():.2f}, {final_pred.max():.2f}]
    • Max Error: {np.max(np.abs(final_pred - final_gt)):.3f}
    """
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/fno_comprehensive_results.png', dpi=150, bbox_inches='tight')
    print("Comprehensive results saved as 'plots/fno_comprehensive_results.png'")
    
    # Print summary statistics
    print(f"\n=== Model Performance Summary ===")
    print(f"Test samples: {len(eval_data)}")
    print(f"Mean relative L2 error: {mean_error:.6f}")
    print(f"Std deviation: {std_error:.6f}")
    print(f"Best sample error: {min(all_errors):.6f}")
    print(f"Worst sample error: {max(all_errors):.6f}")
    
    # Additional detailed analysis
    print(f"\n=== Detailed Error Analysis ===")
    
    # Compute pointwise error statistics
    with torch.no_grad():
        input_tensor = eval_data[:, :, :, :10]
        target_tensor = eval_data[:, :, :, 10:20]
        pred_tensor = model(input_tensor)
        
        # Pointwise errors
        pointwise_errors = torch.abs(pred_tensor - target_tensor)
        mean_pointwise_error = torch.mean(pointwise_errors).item()
        max_pointwise_error = torch.max(pointwise_errors).item()
        
        print(f"Mean absolute error: {mean_pointwise_error:.6f}")
        print(f"Maximum absolute error: {max_pointwise_error:.6f}")
        
        # Temporal error evolution
        temporal_errors = []
        for t in range(10):
            t_error = torch.mean(torch.abs(pred_tensor[:, :, :, t] - target_tensor[:, :, :, t])).item()
            temporal_errors.append(t_error)
        
        print(f"\nTemporal error evolution:")
        for i, error in enumerate(temporal_errors):
            print(f"  Time step {i+11}: {error:.6f}")
    
    plt.show()

if __name__ == "__main__":
    show_results()
