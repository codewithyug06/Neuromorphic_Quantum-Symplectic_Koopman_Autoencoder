import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# --- IMPORT CUSTOM MODULES ---
from nq_skae_data_loader import KSDataset
from main_train import NQ_SKAE_Hybrid, CONFIG

def run_inference():
    print("=== Starting NQ-SKAE Inference & Visualization ===")

    # 1. Force CPU for Inference (Faster for single samples)
    # We update the global config just for this run
    inference_config = CONFIG.copy()
    inference_config["DEVICE"] = torch.device("cpu")
    
    # 2. Locate the Model File
    model_path = "best.pt"
    if not os.path.exists(model_path):
        # Fallback check
        if os.path.exists("nq_skae_best_model.pth"):
            model_path = "nq_skae_best_model.pth"
        else:
            print(f"\n[!] CRITICAL ERROR: Model file '{model_path}' not found.")
            print("    Please run 'main_train.py' first to generate the model.")
            return

    print(f"    >> Loading Model from: {model_path}")
    
    # 3. Load Model
    model = NQ_SKAE_Hybrid(inference_config)
    try:
        # map_location='cpu' ensures it loads even if you trained on GPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"[!] Error loading state dict: {e}")
        return
        
    model.eval()

    # 4. Load Data & Stats
    dataset = KSDataset()
    # Fetch normalization stats to restore real physical values
    data_mean, data_std = dataset.get_stats()
    
    # 5. Select a Test Sample
    # We pick index 500 (mid-dataset) to show established chaos
    sample_idx = 500
    x_t, x_next_true = dataset[sample_idx]
    
    # Add batch dimension (1, 1024)
    x_t_batch = x_t.unsqueeze(0)

    # 6. Predict (Forward Pass)
    print(f"    >> Running Quantum Prediction on Sample #{sample_idx}...")
    with torch.no_grad():
        _, x_pred = model(x_t_batch)

    # 7. Un-Normalize (Convert back to Real Physics Units)
    true_wave_norm = x_next_true.numpy()
    pred_wave_norm = x_pred.squeeze().numpy()
    
    # Formula: Real = (Normalized * Std) + Mean
    true_wave_real = (true_wave_norm * data_std) + data_mean
    pred_wave_real = (pred_wave_norm * data_std) + data_mean

    # 8. Calculate Metrics (MSE on Real Data)
    mse = np.mean((true_wave_real - pred_wave_real)**2)
    print(f"    >> Inference MSE (Real Physics): {mse:.6f}")

    # 9. Plotting (Professional Grade)
    plt.figure(figsize=(12, 6), dpi=300) # High DPI for PPT
    
    # Plot Ground Truth
    plt.plot(true_wave_real, label="Actual Physics (Ground Truth)", 
             color='black', linewidth=2.5, alpha=0.8)
    
    # Plot Prediction
    plt.plot(pred_wave_real, label="NQ-SKAE Prediction (Quantum)", 
             color='#00FFFF', linestyle='--', linewidth=2) # Cyan color
    
    # Styling
    plt.title(f"NQ-SKAE: Chaotic Fluid Prediction (MSE: {mse:.5f})", fontsize=14, fontweight='bold')
    plt.xlabel("Spatial Grid (0-1024)", fontsize=12)
    plt.ylabel("Fluid Amplitude (u)", fontsize=12)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.xlim(0, 1024)
    
    # 10. Save Output
    output_file = "nq_skae_final_result.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"    >> Success! Graph saved to: {os.path.abspath(output_file)}")
    
    # Show plot (optional, block=False allows script to finish)
    plt.show()

if __name__ == "__main__":
    run_inference()