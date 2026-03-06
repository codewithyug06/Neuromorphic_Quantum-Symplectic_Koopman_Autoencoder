import pennylane as qml
import torch
import torch.nn as nn
from typing import List, Tuple

class QuantumKoopmanLayer(nn.Module):
    """
    Module B: The Photonic Quantum Evolver (God-Level Edition).
    
    Features:
    - Hybrid Execution: Automatically offloads Quantum Sim to CPU (fastest) while keeping Gradients alive.
    - Stability Clamping: Prevents 'Squeezing' parameters from exploding (NaN protection).
    - Robust State Management: Handles batch processing without memory leaks.
    """
    
    def __init__(self, num_modes: int = 8, num_layers: int = 3):
        super().__init__()
        self.num_modes = num_modes
        self.num_layers = num_layers
        
        # 1. Device: CPU-only 'default.gaussian' is 10-50x faster for CV simulation than GPU
        self.dev = qml.device("default.gaussian", wires=num_modes)
        
        # 2. Trainable Quantum Parameters (Initialized with low variance for stability)
        # We use a small factor (0.05) to start near the Identity operation
        self.squeezing_r = nn.Parameter(torch.randn(num_layers, num_modes) * 0.05)
        self.squeezing_phi = nn.Parameter(torch.randn(num_layers, num_modes) * 0.1)
        self.rotation_phi = nn.Parameter(torch.randn(num_layers, num_modes) * 0.1)
        self.bs_theta = nn.Parameter(torch.randn(num_layers, num_modes - 1) * 0.1)
        self.bs_phi = nn.Parameter(torch.randn(num_layers, num_modes - 1) * 0.1)

        # 3. Define QNodes (Quantum Circuits)
        self.qnode_x = qml.QNode(self._circuit_x, self.dev, interface="torch")
        self.qnode_p = qml.QNode(self._circuit_p, self.dev, interface="torch")

    def extra_repr(self) -> str:
        return f"num_modes={self.num_modes}, num_layers={self.num_layers}, device='default.gaussian'"

    def _apply_ops(self, inputs, sq_r, sq_phi, rot_phi, bs_theta, bs_phi):
        """Builds the Quantum Circuit dynamically."""
        q_vals = inputs[:self.num_modes]
        p_vals = inputs[self.num_modes:]
        
        # A. Embedding (Classical -> Quantum) via Displacement
        for i in range(self.num_modes):
            # Encode Position (q) -> Real, Momentum (p) -> Imaginary
            alpha = q_vals[i] + 1j * p_vals[i]
            qml.Displacement(torch.abs(alpha), torch.angle(alpha), wires=i)

        # B. Evolution (The Koopman Operator Ansatz)
        for l in range(self.num_layers):
            # 1. Squeezing (Non-linear interactions)
            for i in range(self.num_modes):
                qml.Squeezing(sq_r[l, i], sq_phi[l, i], wires=i)
            
            # 2. Rotation (Phase evolution)
            for i in range(self.num_modes):
                qml.Rotation(rot_phi[l, i], wires=i)
            
            # 3. Beam Splitters (Mode mixing / Entanglement)
            for i in range(self.num_modes - 1):
                qml.Beamsplitter(bs_theta[l, i], bs_phi[l, i], wires=[i, i+1])

    def _circuit_x(self, inputs, sq_r, sq_phi, rot_phi, bs_theta, bs_phi, wire):
        self._apply_ops(inputs, sq_r, sq_phi, rot_phi, bs_theta, bs_phi)
        return qml.expval(qml.QuadX(wire))

    def _circuit_p(self, inputs, sq_r, sq_phi, rot_phi, bs_theta, bs_phi, wire):
        self._apply_ops(inputs, sq_r, sq_phi, rot_phi, bs_theta, bs_phi)
        return qml.expval(qml.QuadP(wire))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust Forward Pass:
        1. Clamps parameters for stability.
        2. Moves data to CPU for Quantum Sim.
        3. Returns results to original device (GPU/CPU).
        """
        original_device = x.device
        
        # --- STABILITY CLAMP ---
        # Prevent "Infinite Energy" crashes by limiting squeezing magnitude
        safe_sq_r = torch.clamp(self.squeezing_r, -1.0, 1.0)
        
        # --- HYBRID TRANSFER ---
        # Move inputs to CPU (Quantum Simulator lives here)
        x_cpu = x.cpu()
        
        # Move weights to CPU for the simulation
        # (We use local variables to avoid messing up the main GPU gradients)
        sq_r_cpu = safe_sq_r.cpu()
        sq_phi_cpu = self.squeezing_phi.cpu()
        rot_phi_cpu = self.rotation_phi.cpu()
        bs_theta_cpu = self.bs_theta.cpu()
        bs_phi_cpu = self.bs_phi.cpu()

        batch_results = []
        
        # --- BATCH EXECUTION ---
        # Loop over batch (PennyLane default.gaussian doesn't natively batch well with Torch)
        for i in range(x.shape[0]):
            curr = x_cpu[i]
            
            # Run Circuit for Position (X) and Momentum (P)
            # We list comprehension over modes for speed
            q_out = [
                self.qnode_x(curr, sq_r_cpu, sq_phi_cpu, rot_phi_cpu, bs_theta_cpu, bs_phi_cpu, m) 
                for m in range(self.num_modes)
            ]
            p_out = [
                self.qnode_p(curr, sq_r_cpu, sq_phi_cpu, rot_phi_cpu, bs_theta_cpu, bs_phi_cpu, m) 
                for m in range(self.num_modes)
            ]
            
            # Stack results: [q1, q2... p1, p2...]
            batch_results.append(torch.cat([torch.stack(q_out), torch.stack(p_out)]))
            
        # Return to original device (e.g., CUDA) for the Decoder to handle
        return torch.stack(batch_results).to(original_device).float()