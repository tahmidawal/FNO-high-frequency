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
from timeit import default_timer

# Add local paths for FNO modules
sys.path.append('.')
sys.path.append('./FNO/PyTorch')

# Import FNO components (matching notebook imports)
try:
    from FNO.fno_2d_time import FNO2DTime
    from FNO.lploss import LpLoss
    from FNO.train import train_model
    print("Successfully imported FNO components from local modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Using alternative imports...")
    
    # Try alternative import paths
    try:
        from FNO.PyTorch.fno_2d_time import FNO2DTime
        from FNO.PyTorch.lploss import LpLoss
        from FNO.PyTorch.train import train_model
        print("Successfully imported from FNO.PyTorch")
    except ImportError as e2:
        print(f"Alternative import failed: {e2}")
        print("Creating simplified FNO2DTime implementation...")
        
        # Simplified FNO2DTime implementation
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
        
        class LpLoss(torch.nn.Module):
            def __init__(self, p=2):
                super(LpLoss, self).__init__()
                self.p = p

            def forward(self, x, y):
                num_examples = x.size()[0]
                diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
                y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
                return torch.sum(diff_norms / y_norms) / num_examples
        
        def train_model(model, train_loader, eval_loader, epochs=100, device='cuda'):
            """Simplified training function matching notebook interface"""
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = LpLoss()
            
            train_losses = []
            eval_losses = []
            
            model = model.to(device)
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Evaluation
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for x, y in eval_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        loss = loss_fn(out, y)
                        eval_loss += loss.item()
                
                eval_loss /= len(eval_loader)
                eval_losses.append(eval_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Eval Loss = {eval_loss:.6f}")
            
            return train_losses, eval_losses


def main():
    """Replicate the notebook training workflow exactly"""
    
    print("=== FNO Training - Notebook Replication ===")
    
    # Configuration (matching notebook)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load our generated data
    data_file = 'ns_data_v1e-5_N100.mat'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found!")
        return
    
    print(f"Loading data from {data_file}...")
    mat_data = scipy.io.loadmat(data_file)
    
    # Extract data (matching notebook format)
    u_data = mat_data['u']  # Shape: (N, 64, 64, 200)
    print(f"Data shape: {u_data.shape}")
    
    # Convert to tensor (matching notebook)
    if isinstance(u_data, np.ndarray):
        u_data = torch.from_numpy(u_data).float()
    
    u_data = u_data.to(device)
    
    # Data preprocessing (exact match to notebook)
    print("Preprocessing data...")
    
    # Split data into train and eval (like notebook: 1000/200, but we have 100)
    train_size = min(80, int(0.8 * u_data.shape[0]))  # Use 80% for training
    train_data = u_data[:train_size, :, :, :]
    eval_data = u_data[train_size:, :, :, :]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Eval data shape: {eval_data.shape}")
    
    # Split data from time 0-10 and 10-20 (exact match to notebook)
    u_train = train_data[:, :, :, :10]  # First 10 time steps
    a_train = train_data[:, :, :, 10:20]  # Next 10 time steps
    
    u_eval = eval_data[:, :, :, :10]
    a_eval = eval_data[:, :, :, 10:20]
    
    print(f"Training input shape: {u_train.shape}")
    print(f"Training target shape: {a_train.shape}")
    print(f"Eval input shape: {u_eval.shape}")
    print(f"Eval target shape: {a_eval.shape}")
    
    # Create data loaders (matching notebook)
    batch_size = 50  # Same as notebook
    train_loader = DataLoader(
        TensorDataset(u_train, a_train),
        batch_size=batch_size, shuffle=True
    )
    
    eval_loader = DataLoader(
        TensorDataset(u_eval, a_eval),
        batch_size=batch_size, shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")
    
    # Define the model (matching notebook)
    print("Creating FNO2DTime model...")
    model = FNO2DTime().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training (matching notebook)
    print("Starting training...")
    epochs = 100  # Same as notebook
    start_time = default_timer()
    
    loss, mse = train_model(model, train_loader, eval_loader, epochs=epochs, device=device)
    
    end_time = default_timer()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Results (matching notebook)
    print("Plotting results...")
    
    # Plot the loss and mse (exact match to notebook)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(mse)
    plt.title('MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/fno_training_results_notebook_replication.png', dpi=150, bbox_inches='tight')
    print("Training results saved as 'plots/fno_training_results_notebook_replication.png'")
    
    # Making predictions (matching notebook)
    print("Making predictions...")
    
    # Test on a sample
    if len(eval_data) > 0:
        data_test = eval_data[:1, :, :, :]  # Take first sample
        print(f"Test data shape: {data_test.shape}")
        
        model.eval()
        with torch.no_grad():
            input_test = data_test[:, :, :, :10]
            target_test = data_test[:, :, :, 10:20]
            
            prediction = model(input_test)
            
            print(f"Prediction shape: {prediction.shape}")
            print(f"Target shape: {target_test.shape}")
            
            # Calculate test loss
            loss_fn = LpLoss()
            test_loss = loss_fn(prediction, target_test)
            print(f"Test loss: {test_loss:.6f}")
            
            # Save model
            torch.save(model.state_dict(), 'fno_model_notebook_replication.pth')
            print("Model saved as 'fno_model_notebook_replication.pth'")
    
    print("\n=== Training Complete ===")
    print(f"Final train loss: {loss[-1]:.6f}")
    print(f"Final eval loss: {mse[-1]:.6f}")
    
    return model, loss, mse

if __name__ == "__main__":
    model, loss, mse = main()
