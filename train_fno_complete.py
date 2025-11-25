#!/usr/bin/env python3

import sys
import os
# Add the PyTorch environment manually
sys.path.append('/cluster/tufts/paralab/tawal01/python310_libs/lib/python3.10/site-packages')

import torch
import math
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

# Add local paths for FNO modules
sys.path.append('.')
sys.path.append('./FNO/PyTorch')

# Import FNO components
try:
    from FNO.PyTorch.fno import FNO
    from losses.lploss import LpLoss
    from training.train import train_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Using simplified FNO implementation...")
    
    # Simplified FNO implementation for testing
    class SimpleFNO(torch.nn.Module):
        def __init__(self, modes1=12, modes2=12, width=64):
            super(SimpleFNO, self).__init__()
            self.width = width
            self.fc0 = torch.nn.Linear(11, width)
            self.conv0 = SpectralConv2d(width, width, modes1, modes2)
            self.conv1 = SpectralConv2d(width, width, modes1, modes2)
            self.conv2 = SpectralConv2d(width, width, modes1, modes2)
            self.conv3 = SpectralConv2d(width, width, modes1, modes2)
            self.w0 = torch.nn.Conv2d(width, width, 1)
            self.w1 = torch.nn.Conv2d(width, width, 1)
            self.w2 = torch.nn.Conv2d(width, width, 1)
            self.w3 = torch.nn.Conv2d(width, width, 1)
            self.fc1 = torch.nn.Linear(width, 128)
            self.fc2 = torch.nn.Linear(128, 1)

        def forward(self, x):
            batchsize = x.shape[0]
            size_x, size_y = x.shape[2], x.shape[3]
            
            x = self.fc0(x)
            x = x.permute(0, 3, 1, 2)
            
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
            
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = torch.nn.functional.relu(x)
            x = self.fc2(x)
            
            return x.squeeze(-1)

    class SpectralConv2d(torch.nn.Module):
        def __init__(self, in_channels, out_channels, modes1, modes2):
            super(SpectralConv2d, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1
            self.modes2 = modes2
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
            self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        def forward(self, x):
            batchsize = x.shape[0]
            ft = torch.fft.rfft2(x, dim=(-2, -1))
            ft = ft[:, :, :self.modes1, :self.modes2]
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", ft, self.weights1)
            out_ft[:, :, :self.modes1, 1:self.modes2] += torch.einsum("bixy,ioxy->boxy", ft.conj(), self.weights2)
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
            return x

    # Use simple FNO
    FNO = SimpleFNO
    
    class LpLoss(torch.nn.Module):
        def __init__(self, p=2):
            super(LpLoss, self).__init__()
            self.p = p

        def forward(self, x, y):
            num_examples = x.size()[0]
            diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
            return torch.sum(diff_norms / y_norms) / num_examples

    def train_model(model, train_loader, eval_loader, epochs=100, lr=0.001, device='cuda'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = LpLoss()
        
        train_losses = []
        eval_losses = []
        
        for epoch in trange(epochs, desc="Training"):
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


# Data generation functions (copied from our working script)
class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size//2
        if dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)
            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers
            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self, N):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real
    

def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    N = w0.size()[-1]
    k_max = math.floor(N/2.0)
    steps = math.ceil(T/delta_t)
    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)
    record_time = math.floor(steps/record_steps)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    k_x = k_y.transpose(0,1)
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)
    c = 0
    t = 0.0
    for j in range(steps):
        psi_h = w_h / lap
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))
        F_h = torch.fft.rfft2(q*w_x + v*w_y)
        F_h = dealias* F_h
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)
        t += delta_t
        if (j+1) % record_time == 0:
            w = torch.fft.irfft2(w_h, s=(N, N))
            sol[...,c] = w
            sol_t[c] = t
            c += 1
    return sol, sol_t


def generate_ns_data(resolution, N, f, visc, delta_t, T_final, record_steps, batch_size, device, debug=False):
    c = 0
    t0 = default_timer()
    GRF = GaussianRF(2, resolution, alpha=2.5, tau=7, device=device)
    a = torch.zeros(N, resolution, resolution)
    u = torch.zeros(N, resolution, resolution, record_steps)
    total_batches = N // batch_size
    for j in range(total_batches):
        w0 = GRF.sample(batch_size)
        sol, solt_t = navier_stokes_2d(w0, f, visc, T_final, delta_t, record_steps)
        a[c:(c+batch_size),...] = w0
        u[c:(c+batch_size),...] = sol
        c += batch_size
        t1 = default_timer()
        if debug:
            print(f"Batch {j+1}/{total_batches} | N: {c}/{N} | Time: {t1-t0:.2f} s")
    return a.cpu(), u.cpu(), solt_t.cpu()


def main():
    """Complete FNO training pipeline"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data generation parameters
    resolution = 64
    N = 100  # Generate 100 samples for testing
    visc = 1e-5  # Viscosity coefficient
    delta_t = 1e-4  # Time step
    T_final = 20.0  # Final time
    record_steps = 200  # Number of time steps to record
    batch_size = 10  # Batch size for generation
    
    # Training parameters
    train_split = 80  # Number of samples for training
    time_input_steps = 10  # Use first 10 time steps as input
    time_output_steps = 10  # Predict next 10 time steps
    train_batch_size = 20
    epochs = 50
    lr = 0.001
    
    print("=== Data Generation ===")
    
    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, resolution+1, device=device)
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    print(f"Generating {N} Navier-Stokes samples...")
    
    # Generate data
    start_time = default_timer()
    a, u, sol_t = generate_ns_data(
        resolution=resolution,
        N=N,
        f=f,
        visc=visc,
        delta_t=delta_t,
        T_final=T_final,
        record_steps=record_steps,
        batch_size=batch_size,
        device=device,
        debug=True
    )
    
    end_time = default_timer()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds")
    print(f"Data shapes: a={a.shape}, u={u.shape}, t={sol_t.shape}")
    
    # Save data
    output_file = 'ns_data_v1e-5_N100.mat'
    print(f"Saving data to {output_file}...")
    scipy.io.savemat(output_file, mdict={
        'a': a.numpy(), 
        'u': u.numpy(), 
        't': sol_t.numpy()
    })
    print("Data saved successfully!")
    
    print("\n=== Data Preprocessing ===")
    
    # Split data into train and eval
    train_data = u[:train_split, :, :, :]
    eval_data = u[train_split:, :, :, :]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Eval data shape: {eval_data.shape}")
    
    # Split time dimension: use first 10 steps as input, next 10 as target
    u_train = train_data[:, :, :, :time_input_steps]
    a_train = train_data[:, :, :, time_input_steps:time_input_steps+time_output_steps]
    
    u_eval = eval_data[:, :, :, :time_input_steps]
    a_eval = eval_data[:, :, :, time_input_steps:time_input_steps+time_output_steps]
    
    print(f"Training input shape: {u_train.shape}")
    print(f"Training target shape: {a_train.shape}")
    print(f"Eval input shape: {u_eval.shape}")
    print(f"Eval target shape: {a_eval.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(u_train, a_train)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    eval_dataset = TensorDataset(u_eval, a_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=train_batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")
    
    print("\n=== Model Training ===")
    
    # Create model with correct parameters
    model = FNO(
        modes=[8, 8, 6],
        num_fourier_layers=8,
        in_channels=13,  # 10 time steps + 3 grid coordinates
        lifting_channels=128,
        projection_channels=128,
        mid_channels=64,
        out_channels=10,  # Predict 10 time steps
        activation=torch.nn.GELU(),
        padding=(0, 0, 3)
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_losses, eval_losses = train_model(
        model, train_loader, eval_loader, 
        epochs=epochs, lr=lr, device=device
    )
    
    print("\n=== Training Complete ===")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final eval loss: {eval_losses[-1]:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FNO Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('fno_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved as 'fno_training_curves.png'")
    
    # Test on a sample
    print("\n=== Testing Model ===")
    model.eval()
    with torch.no_grad():
        x_test, y_test = next(iter(eval_loader))
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        pred = model(x_test)
        loss_fn = LpLoss()
        test_loss = loss_fn(pred, y_test)
        
        print(f"Test sample loss: {test_loss:.6f}")
        print(f"Input shape: {x_test.shape}")
        print(f"Target shape: {y_test.shape}")
        print(f"Prediction shape: {pred.shape}")
    
    print("\n=== Complete Pipeline Finished ===")
    
    return model, train_losses, eval_losses


if __name__ == "__main__":
    model, train_losses, eval_losses = main()
