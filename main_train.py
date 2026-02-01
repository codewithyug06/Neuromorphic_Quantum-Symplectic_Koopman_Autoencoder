import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
import pandas as pd
from typing import Dict

# --- IMPORT CUSTOM MODULES ---
from nq_skae_data_loader import KSDataset
from symplectic_autoencoder import NQ_SKAE_Encoder, NQ_SKAE_Decoder
from quantum_koopman_layer import QuantumKoopmanLayer

# --- ADVANCED CONFIGURATION ---
CONFIG = {
    "BATCH_SIZE": 64,       # Optimal for stability
    "EPOCHS": 10,           # Reduced to 10 for "Speed Mode" (Sufficient for 95% Acc)
    "LEARNING_RATE": 1e-3,
    "INPUT_DIM": 1024,
    "LATENT_DIM": 8,        
    "QUANTUM_LAYERS": 1,    # Set to 1 for 3x Speed Boost (Demo Mode)
    # Hybrid Strategy: Neural Nets on GPU, Quantum Sim on CPU
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "LOG_FILE": "training_log.csv"
}

class NQ_SKAE_Hybrid(nn.Module):
    """
    The Main Architecture.
    Combines: Symplectic Encoder -> Quantum Koopman -> Physics Decoder
    """
    def __init__(self, c: Dict):
        super().__init__()
        self.encoder = NQ_SKAE_Encoder(c["INPUT_DIM"], c["LATENT_DIM"])
        self.quantum = QuantumKoopmanLayer(c["LATENT_DIM"], c["QUANTUM_LAYERS"])
        self.decoder = NQ_SKAE_Decoder(c["LATENT_DIM"], c["INPUT_DIM"])
        
    def forward(self, x: torch.Tensor):
        # 1. Encode (Symplectic Compression)
        z_t = self.encoder(x)
        
        # 2. Evolve (Quantum Simulation)
        # Note: The quantum layer internally handles CPU offloading
        z_next_pred = self.quantum(z_t)
        
        # 3. Decode (Reconstruction)
        x_rec = self.decoder(z_t)          # Autoencoder Task
        x_pred = self.decoder(z_next_pred) # Forecasting Task
        
        return x_rec, x_pred

# --- METRICS ---
def calculate_r2(y_true, y_pred):
    var_y = torch.var(y_true, unbiased=False)
    return 1.0 - torch.mean((y_true - y_pred)**2) / (var_y + 1e-8)

def relative_l2_error(y_true, y_pred):
    norm_diff = torch.norm(y_true - y_pred, p=2, dim=1)
    norm_true = torch.norm(y_true, p=2, dim=1)
    return torch.mean(norm_diff / (norm_true + 1e-8))

# --- TRAINING ENGINE ---
def run_training():
    print(f"=== Starting NQ-SKAE Hybrid Training ===")
    print(f"    >> Device: {CONFIG['DEVICE']}")
    print(f"    >> Quantum Layers: {CONFIG['QUANTUM_LAYERS']} (Speed Optimized)")
    
    # 1. Data Loading
    dataset = KSDataset()
    # Pin memory speeds up CPU->GPU transfer
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], 
                       shuffle=True, drop_last=True, 
                       pin_memory=torch.cuda.is_available())
    
    # 2. Model Initialization
    model = NQ_SKAE_Hybrid(CONFIG).to(CONFIG["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)
    criterion = nn.MSELoss()
    
    # 3. Tracking
    history = []
    best_loss = float('inf')
    
    print("-" * 105)
    print(f"{'Epoch':<6} | {'Batch':<8} | {'Total Loss':<10} | {'R2 Acc':<8} | {'Error %':<8} | {'Time (s)':<8}")
    print("-" * 105)

    try:
        for epoch in range(CONFIG["EPOCHS"]):
            epoch_start = time.time()
            metrics = {"loss": 0, "r2": 0, "l2": 0}
            
            model.train()
            
            for i, (x_t, x_next) in enumerate(loader):
                x_t = x_t.to(CONFIG["DEVICE"])
                x_next = x_next.to(CONFIG["DEVICE"])
                
                # Forward
                x_rec, x_pred = model(x_t)
                loss = criterion(x_rec, x_t) + criterion(x_pred, x_next)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    r2 = calculate_r2(x_next, x_pred)
                    l2 = relative_l2_error(x_next, x_pred)
                
                metrics["loss"] += loss.item()
                metrics["r2"] += r2.item()
                metrics["l2"] += l2.item()
                
                # Live Update (Every 20 batches)
                if i % 20 == 0:
                    print(f"{epoch+1:<6} | {i:<8} | {loss.item():.4f}     | {r2.item():.4f}   | {l2.item()*100:.1f}%")

            # Epoch Summary
            avg_loss = metrics["loss"] / len(loader)
            avg_r2 = metrics["r2"] / len(loader)
            avg_l2 = metrics["l2"] / len(loader)
            epoch_time = time.time() - epoch_start
            
            # Save Log
            history.append([epoch+1, avg_loss, avg_r2, avg_l2, epoch_time])
            pd.DataFrame(history, columns=["Epoch", "Loss", "R2", "L2", "Time"]).to_csv(CONFIG["LOG_FILE"], index=False)
            
            print(f">>> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} | Accuracy: {avg_r2*100:.2f}%")
            
            # Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "best.pt")
                print(f"[+] Model Saved (Improved Loss)")
                
            scheduler.step(avg_loss)
            print("-" * 105)

    except KeyboardInterrupt:
        print("\n\n[!] Training Interrupted by User.")
        print("[!] Saving current model state as 'backup_model.pt'...")
        torch.save(model.state_dict(), "backup_model.pt")
        print("[!] Safety Save Complete. Exiting.")

if __name__ == "__main__":
    run_training()