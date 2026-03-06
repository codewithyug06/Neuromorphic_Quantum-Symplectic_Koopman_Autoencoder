# nq_skae/__init__.py

from .data import KSDataset
from .models import NQ_SKAE_Encoder, NQ_SKAE_Decoder, SymplecticLinear
from .quantum import QuantumKoopmanLayer

# Version of the library
__version__ = "0.1.0"

# What gets imported when someone types 'from nq_skae import *'
__all__ = [
    "KSDataset",
    "NQ_SKAE_Encoder", 
    "NQ_SKAE_Decoder",
    "SymplecticLinear",
    "QuantumKoopmanLayer"
]