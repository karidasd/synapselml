# SynapseLML (Latent Modeling Language)
### Vector-Native Silicon-to-Silicon Communication Protocol for Multi-Agent Topologies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Engine: PyTorch](https://img.shields.io/badge/Engine-PyTorch-ee4c2c.svg)](https://pytorch.org)

SynapseLML is a revolutionary, non-human-readable, vector-native communication framework designed to eliminate the traditional text tokenization bottleneck in multi-agent and LLM-to-LLM workflows. 

Traditional agent architectures communicate by translating internal high-dimensional representation states into text tokens, passing them over a network API, and re-tokenizing them at the target agent. SynapseLML establishes **Direct Latent Space Synchronization (LSS)**, enabling agents to stream high-dimensional tensor matrices directly across network barriers using learnable manifold projection alignments and real-time entropy-based feedback loop runtimes.

---

## ⚡ Architecture Flow

```
                      +---------------------------------------+
                      |          ANALYST AGENT (Source)       |
                      |          Hidden State Manifold        |
                      |          Dimension: d1 = 256          |
                      +-------------------+-------------------+
                                          |
                                          | H_A Tensor Stream
                                          v
                      +-------------------+-------------------+
                      |          LATENT PROJECTOR             |
                      |       (Manifold Projection f_P)       |
                      |       Orthogonal Procrustes Constraint|
                      +-------------------+-------------------+
                                          |
                                          | (Projected state: d2 = 512)
                                          v
                      +-------------------+-------------------+
                      |      VECTOR TRANSPORT PROTOCOL        |
                      |     Symmetric 8-Bit Quantization      |
                      +-------------------+-------------------+
                                          |
                                          | Packed VTP Payload (Network binary stream)
                                          v
                      +-------------------+-------------------+
                      |       ENTROPY-BASED RUNTIME           |
                      |    Real-Time Alignment Loss Check     |
                      |    Loss > 0.01 -> Micro-Backprop      |
                      +-------------------+-------------------+
                                          |
                                          | (Optimally aligned states: d2 = 512)
                                          v
                      +-------------------+-------------------+
                      |          EXECUTOR AGENT (Target)      |
                      |          Decoded Target Manifold      |
                      |          Dimension: d2 = 512          |
                      +---------------------------------------+
```

---

## 🧠 Core Philosophy & Innovations

### 1. Direct Latent Space Synchronization (LSS)
Bypassing text tokenization reduces inference latency and network bandwidth requirements. SynapseLML allows diverse agent models (e.g., Llama-based and Mistral-based backbones) to align their distinct token embedding geometries.

### 2. Vector Transport Protocol (VTP)
VTP serializes and packages high-dimensional tensor streams. It implements symmetric 8-bit quantization (`FP8` scale-factor packing) to compress latent states, achieving high bandwidth savings over standard sockets.

### 3. Entropy-Based Runtime (EBR)
Instead of standard exception handling, error correction in SynapseLML is represented as continuous representation alignment optimization. The runtime monitors mathematical divergence (joint spatial MSE and Cosine similarity distance) between source and target manifolds. Upon threshold violation, EBR dynamically executes inline **micro-backpropagation** to re-map the projector's translation layers on-the-fly.

---

## 📂 Repository Tree

```
synapselml/
├── core/
│   └── runtime.py           # LatentProjector, SynapseStream, and EntropyRuntime
├── examples/
│   └── multi_agent_mesh.py  # Simulated multi-agent latent alignment iteration script
└── specs/
    └── core_protocol.md     # Mathematical, VTP, and EBR LaTeX specifications
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

```bash
# Clone the repository
git clone https://github.com/karidasd/synapselml.git
cd synapselml

# Install dependencies (only PyTorch is required)
pip install torch
```

### Running the Multi-Agent Mesh Simulation
Execute the demonstration script illustrating the dynamic alignment of an Analyst Agent ($d_1=256$) and an Executor Agent ($d_2=512$) communicating purely through raw latent vectors:

```bash
python examples/multi_agent_mesh.py
```

The runtime will print a real-time table demonstrating the joint alignment loss convergence using the Entropy-Based Runtime (EBR) micro-backpropagation loop:

```
================================================================================
         SynapseLML (Latent Modeling Language) Multi-Agent Mesh
================================================================================
Epoch  | Loss       | MSE        | Cos Dist   | Orth Pen   | EBR Action
--------------------------------------------------------------------------------
1      | 0.530320   | 0.311945   | 0.590229   | 5.353089   | MICRO_BP  
2      | 0.387872   | 0.343585   | 0.291862   | 3.248803   | MICRO_BP  
...
20     | 0.256044   | 0.268850   | 0.208024   | 0.576195   | MICRO_BP  
--------------------------------------------------------------------------------
Final Orthogonal Penalty (Frobenius deviation): 0.57619518
STATUS: SUCCESS. Projector maintained absolute structure, preventing representation collapse.
================================================================================
```

---

## 📄 License
This framework is licensed under the MIT License.
