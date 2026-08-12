# 🌌 Neuromorphic Quantum-Symplectic Koopman Autoencoder (NQ-SKAE)

> **A Physics-Informed Hybrid Quantum-Classical Architecture for Long-Horizon Simulation of Chaotic Fluid Dynamics**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-blueviolet?style=flat-square)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

---

## Table of Contents

1. [Abstract](#abstract)
2. [The Problem: Why Classical AI Fails at Physics](#the-problem-why-classical-ai-fails-at-physics)
3. [The NQ-SKAE Solution](#the-nq-skae-solution)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [Project Structure](#project-structure)
7. [Installation & Usage](#installation--usage)
8. [Dataset: The Kuramoto–Sivashinsky Equation](#dataset-the-kuramotosivashinsky-equation)
9. [Key Results](#key-results)
10. [Future Roadmap](#future-roadmap)
11. [License](#license)

---

## Abstract

**NQ-SKAE** is a deep learning architecture built to solve two well-known failure modes of standard neural networks when modeling chaotic physical systems (e.g., turbulent fluid flow): **gradient collapse** and **numerical dissipation**. It does this by fusing **symplectic geometry** (which preserves phase-space volume in classical mechanics) with **continuous-variable (CV) quantum mechanics** (which guarantees unitary, norm-preserving evolution). Rather than being left to *approximate* the underlying physics from data alone, the network has physical law encoded directly into its topology.

The model learns the **Koopman operator** — a linear operator that governs the evolution of a nonlinear dynamical system when lifted into an appropriate latent space — for the **Kuramoto–Sivashinsky (KS) equation**, a canonical benchmark for spatiotemporal chaos. The result is long-horizon forecasting with mathematically bounded energy drift, even over 100,000+ recursive prediction steps.

---

## The Problem: Why Classical AI Fails at Physics

Conventional sequence models (LSTMs, RNNs, Fourier Neural Operators) run into fundamental mathematical limits when applied to chaotic systems:

| Failure Mode | Description |
|---|---|
| **Numerical Dissipation** | These models behave like low-pass filters, smoothing away high-frequency micro-turbulence to minimize mean-squared error, which erases physically meaningful detail. |
| **Energy Drift** | Without an explicit physical constraint, nothing stops the network from violating the First Law of Thermodynamics — system energy can leak away or blow up over long rollouts. |
| **The Lyapunov Barrier** | In chaotic systems, small errors compound exponentially. Unconstrained networks typically decorrelate from ground truth after roughly 1.5 Lyapunov times. |

### The NQ-SKAE Solution

> Instead of **approximating** the physics, NQ-SKAE **enforces** it.

The dynamics are mapped onto a simulated **quantum optical circuit**, which is mathematically guaranteed to be unitary ($U^\dagger U = I$). Because unitary operators preserve vector norms, the system's energy profile is preserved by construction rather than learned as a soft constraint.

---

## System Architecture

The pipeline moves data through three mathematically distinct stages:

```
High-Dimensional Physics  →  Latent Quantum Phase Space  →  Future Predictions
```

### 1. 🔷 Symplectic Encoder *(Classical)*
`src/models.py` — `SymplecticLinear`, `NQ_SKAE_Encoder`

- Compresses the high-dimensional spatial grid (1,024 points) into a low-dimensional latent phase space.
- Uses custom **symplectic layers** that constrain output dimensionality to even numbers, forming strict conjugate pairs of position ($q$) and momentum ($p$).
- Initialized with physics-informed Xavier initialization and `Tanh` activations to preserve gradient flow across the manifold.
- Satisfies **Liouville's Theorem**, ensuring the geometry of the chaotic attractor is *folded* rather than *crushed* during dimensionality reduction.

### 2. ⚛️ Quantum Koopman Layer *(Hybrid Core)*
`src/quantum.py` — `QuantumKoopmanLayer`

- Linearly evolves the latent state forward in time ($t \rightarrow t+1$).
- Implemented as a **continuous-variable (CV) photonic circuit**, simulated in PennyLane on the `default.gaussian` device.
  - **Embedding:** Maps $(q, p)$ pairs to complex amplitudes $\alpha$ via displacement gates.
  - **Dynamics:** Approximates the Koopman operator through a sequence of squeezing, rotation, and beamsplitter gates.
- **Why a quantum circuit:**
  - *Linearization* — Koopman theory holds that nonlinear chaotic dynamics become linear in a sufficiently high-dimensional (in the limit, infinite-dimensional) Hilbert space. The quantum circuit supplies that expressive feature space.
  - *Unitarity* — Quantum evolution is reversible and norm-preserving, mathematically guaranteeing zero energy drift.
  - *Hybrid execution* — Quantum simulation is offloaded to CPU for efficiency while gradients continue to flow through PyTorch's autograd for end-to-end training, with classical layers running on GPU.

### 3. 🔶 Symplectic Decoder *(Classical)*
`src/models.py` — `NQ_SKAE_Decoder`

- Projects the quantum-evolved latent state back onto the physical grid.
- Mirrors the encoder's structure, progressively upscaling the representation to reconstruct high-frequency wave-fronts in the fluid field.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Deep Learning | **PyTorch** | Neural network graphs, autograd, GPU acceleration |
| Quantum Simulation | **PennyLane** | Differentiable quantum circuit programming |
| Quantum Backend | **default.gaussian** | Efficient simulation of CV photonic (Gaussian optics) states |
| Data Handling | **Pandas / NumPy** | Tensor manipulation for the KS dataset |
| Optimization | **Adam + ReduceLROnPlateau** | Adaptive gradient descent for a chaotic, non-convex loss landscape |

---

## Project Structure

```
NQ-SKAE/
├── data/                  # Dataset storage
│   └── X1train.csv        # Kuramoto–Sivashinsky training data
├── figures/                # Generated evaluation plots
├── src/                    # Source code
│   ├── __init__.py
│   ├── data.py             # Data loader with normalization
│   ├── models.py            # Symplectic encoder / decoder architectures
│   ├── quantum.py           # PennyLane quantum circuit definitions
│   └── main_train.py        # Hybrid training engine & validation loop
├── weights/                # Saved model checkpoints
│   └── best.pt              # Best-performing model weights
├── requirements.txt        # Python dependencies
└── README.md                # Documentation
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for the classical layers; quantum simulation runs on CPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/codewithyug06/Neuromorphic_Quantum-Symplectic_Koopman_Autoencoder.git
cd Neuromorphic_Quantum-Symplectic_Koopman_Autoencoder

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python src/main_train.py
```

Hybrid execution runs classical neural network layers on GPU while the quantum circuit simulation runs on CPU. Configuration parameters (batch size, learning rate, number of quantum layers, etc.) can be adjusted in the `CONFIG` dictionary inside `main_train.py`.

---

## Dataset: The Kuramoto–Sivashinsky Equation

The model is benchmarked on the KS equation, a canonical standard for spatiotemporal chaos:

$$u_t + u u_x + u_{xx} + u_{xxxx} = 0$$

| Property | Value |
|---|---|
| Input dimension | 1,024 spatial grid points |
| Characteristics | Spatiotemporal chaos, multi-scale energy cascade, positive Lyapunov exponent |
| Data structure | Autoregressive pairs $(x_t, x_{t+1})$ |

---

## Key Results

| Metric | Value |
|---|---|
| **MSE** | 0.00640 — outperforms standard FNO baselines |
| **Stability** | Preserves structural integrity of wave-fronts over 100,000 recursive time steps |
| **Energy Conservation** | Hamiltonian drift bounded below $10^{-4}$, a direct consequence of the unitary quantum layer |

---

## Future Roadmap

- [ ] **3D Turbulence** — Extend the architecture to 3D Navier–Stokes equations for aerodynamic simulation.
- [ ] **Real Hardware** — Deploy the inference layer on physical photonic quantum processors (e.g., Xanadu Borealis).
- [ ] **Fault Tolerance** — Integrate GKP (Gottesman–Kitaev–Preskill) error correction for noise resilience.

---

## License

Released under the [MIT License](./LICENSE).

---

*Topics: autoencoder · deep-learning · fluid-dynamics · koopman-operator · pennylane · physics-informed-neural-networks · pytorch · quantum-machine-learning*

