#!/usr/bin/env python3

import sys
import os
# Add the PyTorch environment manually
sys.path.append('/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages')

import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def visualize_data():
    """Load and visualize the generated Navier-Stokes data"""
    
    # Load the data file
    data_file = 'ns_data_v1e-5_N100.mat'
    
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        print("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.mat'):
                print(f"  - {f}")
        return
    
    print(f"Loading data from {data_file}...")
    mat_data = scipy.io.loadmat(data_file)
    
    # Extract data
    a = mat_data['a']  # Initial conditions
    u = mat_data['u']  # Solutions
    t = mat_data['t']  # Time points
    
    # Fix time points shape - it's (1, 200) but should be (200,)
    if t.shape[0] == 1:
        t = t.flatten()
    elif t.shape[1] == 1:
        t = t.flatten()
    
    print(f"Data shapes:")
    print(f"  - Initial conditions (a): {a.shape}")
    print(f"  - Solutions (u): {u.shape}")
    print(f"  - Time points (t): {t.shape}")
    print(f"  - Time range: {t[0]:.2f} to {t[-1]:.2f}")
    
    # Select a sample to visualize
    sample_idx = 0
    print(f"\nVisualizing sample {sample_idx}...")
    
    # Extract sample data
    initial_vorticity = a[sample_idx, :, :]
    solution_vorticity = u[sample_idx, :, :, :]
    time_points = t  # Already flattened above
    
    print(f"Sample statistics:")
    print(f"  - Initial vorticity range: [{initial_vorticity.min():.3f}, {initial_vorticity.max():.3f}]")
    print(f"  - Final vorticity range: [{solution_vorticity[:, :, -1].min():.3f}, {solution_vorticity[:, :, -1].max():.3f}]")
    print(f"  - Max absolute vorticity: {np.abs(solution_vorticity).max():.3f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Time indices to visualize
    time_indices = [0, 20, 40, 80, 120, 160, 199]
    
    # Initial condition
    im0 = axes[0, 0].imshow(initial_vorticity, cmap='RdBu_r', origin='lower')
    axes[0, 0].set_title(f'Initial Vorticity (t=0)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Evolution at different time points
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
    
    for i, (time_idx, (row, col)) in enumerate(zip(time_indices[1:], positions)):
        im = axes[row, col].imshow(solution_vorticity[:, :, time_idx], cmap='RdBu_r', origin='lower')
        axes[row, col].set_title(f'Vorticity at t={time_points[time_idx]:.1f}')
        axes[row, col].set_xlabel('x')
        if col == 0:
            axes[row, col].set_ylabel('y')
        plt.colorbar(im, ax=axes[row, col])
    
    # Time series at different spatial points
    axes[2, 1].plot(time_points, solution_vorticity[32, 32, :], label='Center (32,32)')
    axes[2, 1].plot(time_points, solution_vorticity[16, 16, :], label='Point (16,16)')
    axes[2, 1].plot(time_points, solution_vorticity[48, 48, :], label='Point (48,48)')
    axes[2, 1].set_title('Vorticity Evolution at Different Points')
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Vorticity')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    # Energy evolution
    energy = np.sum(solution_vorticity**2, axis=(0, 1))
    axes[2, 2].plot(time_points, energy)
    axes[2, 2].set_title('Total Energy Evolution')
    axes[2, 2].set_xlabel('Time')
    axes[2, 2].set_ylabel('Total Energy (∑w²)')
    axes[2, 2].grid(True)
    
    # Maximum absolute vorticity evolution
    max_vorticity = np.max(np.abs(solution_vorticity), axis=(0, 1))
    axes[2, 3].plot(time_points, max_vorticity)
    axes[2, 3].set_title('Max |Vorticity| Evolution')
    axes[2, 3].set_xlabel('Time')
    axes[2, 3].set_ylabel('Max |Vorticity|')
    axes[2, 3].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/navier_stokes_data_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'plots/navier_stokes_data_visualization.png'")
    
    # Additional analysis: Check data for training
    print(f"\n=== Training Data Analysis ===")
    
    # Check the time split for training
    time_input_steps = 10
    time_output_steps = 10
    
    input_data = solution_vorticity[:, :, :time_input_steps]
    target_data = solution_vorticity[:, :, time_input_steps:time_input_steps+time_output_steps]
    
    print(f"Training input shape: {input_data.shape}")
    print(f"Training target shape: {target_data.shape}")
    print(f"Input time range: {time_points[0]:.1f} to {time_points[time_input_steps-1]:.1f}")
    print(f"Target time range: {time_points[time_input_steps]:.1f} to {time_points[time_input_steps+time_output_steps-1]:.1f}")
    
    # Check for NaN or infinite values
    print(f"\n=== Data Quality Check ===")
    print(f"NaN values in data: {np.isnan(u).sum()}")
    print(f"Infinite values in data: {np.isinf(u).sum()}")
    print(f"Zero values in data: {(u == 0).sum()}")
    
    # Check data range
    print(f"Data range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"Mean: {u.mean():.6f}")
    print(f"Std: {u.std():.6f}")
    
    plt.show()

if __name__ == "__main__":
    visualize_data()
