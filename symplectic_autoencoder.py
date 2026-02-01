import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple

class SymplecticLinear(nn.Module):
    """
    Custom Linear Layer meant for Symplectic/Hamiltonian Systems.
    
    Structure:
    - Enforces even dimensions (q, p pairs) required for Liouville's Theorem.
    - Uses Physics-Informed initialization to preserve gradient flow through Tanh activations.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # CRITICAL CHECK: Symplectic geometry requires pairs of (Position, Momentum)
        assert in_features % 2 == 0, f"Input features ({in_features}) must be even for Symplectic Manifolds."
        assert out_features % 2 == 0, f"Output features ({out_features}) must be even for Symplectic Manifolds."
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self._init_layer()

    def _init_layer(self):
        # Xavier Initialization is optimal for Tanh/Sigmoid physics networks
        init.xavier_uniform_(self.linear.weight, gain=init.calculate_gain('tanh'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class NQ_SKAE_Encoder(nn.Module):
    """
    The Symplectic Encoder: Compresses High-Dim Chaos -> Quantum Latent Space.
    
    Architecture:
    - Input: 1024 spatial points (Physical Space)
    - Output: 16 Latent Variables (8 Position 'q', 8 Momentum 'p')
    - Physics: Preserves phase space information for the Quantum Layer.
    """
    def __init__(self, input_dim: int = 1024, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Layer 1: Manifold Reduction (Standard Linear)
        self.pre_process = nn.Linear(input_dim, 64)
        self.act1 = nn.Tanh() # Tanh is preferred in Physics for smooth derivatives
        
        # Layer 2: Symplectic Compression (Enforcing q,p structure)
        self.symp_layer1 = SymplecticLinear(64, 32)
        self.act2 = nn.Tanh()
        
        # Layer 3: Latent Projection
        # Output is latent_dim * 2 because we need pairs (q, p)
        self.symp_layer2 = SymplecticLinear(32, latent_dim * 2) 
        
        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        # Custom initialization to keep gradients stable in deep networks
        init.xavier_uniform_(self.pre_process.weight, gain=init.calculate_gain('tanh'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safety: Ensure input is flattened (Batch, Input_Dim)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = self.act1(self.pre_process(x))
        x = self.act2(self.symp_layer1(x))
        z = self.symp_layer2(x)
        return z

class NQ_SKAE_Decoder(nn.Module):
    """
    The Physics Decoder: Reconstructs the Fluid State.
    
    Function:
    - Projects the Quantum-Evolved Latent States (q, p) back to Physical Grid (1024).
    - Uses progressive upscaling to maintain high-frequency ripples.
    """
    def __init__(self, latent_dim: int = 8, output_dim: int = 1024):
        super().__init__()
        
        # We use a Sequential block for cleaner architecture visualization
        self.net = nn.Sequential(
            # Input: 16 vars (8q + 8p)
            nn.Linear(latent_dim * 2, 64),
            nn.Tanh(),
            
            # Hidden Expansion
            nn.Linear(64, 256),
            nn.Tanh(),
            
            # Final Projection to Physical Grid
            nn.Linear(256, output_dim) 
            # Note: No activation at the end (Linear Reconstruction) is standard for Regression
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('tanh'))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)