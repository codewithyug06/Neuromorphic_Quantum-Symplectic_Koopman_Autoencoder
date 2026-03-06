import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

# --- CONFIGURATION ---
# Primary Path: Your specific local path (Preserved for your workflow)
USER_PATH = Path(r"C:\Users\Yugendhar S\Downloads\Maths\dataset\Kuramoto-Sivashinsky (KS) Datase\csv\train")

class KSDataset(Dataset):
    """
    Production-Grade Dataset Loader for the NQ-SKAE Project.
    """
    
    def __init__(self, directory_path: Optional[str] = None, file_name: str = "X1train.csv"):
        # 1. Smart Path Resolution
        if directory_path:
            self.data_dir = Path(directory_path)
        elif USER_PATH.exists():
            self.data_dir = USER_PATH
        else:
            # Fallback: Check if dataset is in the same folder as this script
            self.data_dir = Path(__file__).parent
            
        self.file_path = self.data_dir / file_name

        print(f"\n[Data Loader] Initializing...")
        
        # 2. File Verification
        if not self.file_path.exists():
             raise FileNotFoundError(
                f"CRITICAL ERROR: Dataset '{file_name}' not found.\n"
                f"Checked location: {self.file_path}\n"
                f"Action: Please move 'X1train.csv' to the script folder or update USER_PATH."
            )

        # 3. Load Data Efficiently
        try:
            # Load as float32 immediately to save RAM and match PyTorch defaults
            df = pd.read_csv(self.file_path, header=None)
            self.data = df.values.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

        # 4. Global Normalization (Crucial for Quantum Embedding)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        
        # Apply Normalization: (X - mu) / sigma
        self.data = (self.data - self.mean) / (self.std + 1e-8)

        print(f"[Data Loader] SUCCESS. Data Loaded & Normalized.")
        print(f"    >> Source: {self.file_path}")
        print(f"    >> Shape: {self.data.shape} (Time Steps, Spatial Points)")

    def __len__(self) -> int:
        return len(self.data) - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = self.data[idx]
        x_next = self.data[idx + 1]
        return torch.from_numpy(x_t), torch.from_numpy(x_next)

    def get_stats(self):
        """Returns (mean, std) so the visualization script can un-normalize the plots."""
        return self.mean, self.std

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    try:
        ds = KSDataset()
        x, y = ds[0]
        print(f"\n[Self-Test] Passed! Output Tensor Shape: {x.shape}")
    except Exception as e:
        print(f"\n[Self-Test] Failed: {e}")