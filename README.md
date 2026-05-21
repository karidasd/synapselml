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
│   └── multi_agent_mesh.py  # Simulated in-memory multi-agent latent alignment script
├── network/
│   ├── client.py            # Async TCP Analyst Client (Source Space, d1=256)
│   └── server.py            # Async TCP Executor Server (Target Space, d2=512)
├── specs/
│   └── core_protocol.md     # Mathematical, VTP, and EBR LaTeX specifications
└── viz/
    └── dashboard.py         # Streamlit & Plotly 3D Real-time Point Cloud Dashboard
```

---

## 🚀 Quick Start & Installation

### Prerequisites
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install torch numpy scikit-learn streamlit plotly pandas
```

### Running the In-Memory Multi-Agent Mesh Simulation
Execute the local simulation script showing convergence of representation spaces in-memory:
```bash
python examples/multi_agent_mesh.py
```

---

## 🌐 Real-World Distributed Network Mesh & 3D Visualization

SynapseLML supports multi-process execution where the Analyst Agent and Executor Agent run in completely isolated OS processes, streaming quantized representations over network sockets and updating a 3D visualization dashboard in real-time.

```
                      +--------------------------+
                      |  Analyst Client Process  |
                      |  (network/client.py)     |
                      +------------+-------------+
                                   |
                                   | Quantized FP8 Stream (VTP)
                                   v
                      +------------+-------------+     Atomic      +-------------------------+
                      |  Executor Server Process |    Telemetry    |   Streamlit Dashboard   |
                      |  (network/server.py)     +================>|   (viz/dashboard.py)    |
                      +--------------------------+  (Shared PKL)   +-------------------------+
```

### 🛰️ The Binary Protocol Specification

1. **VTP Packet Header (32 bytes)**:
   - `Magic Bytes (4 bytes)`: `VTP\x00`
   - `Payload Size (4 bytes)`: Size in bytes of the following serialized quantized tensor array (unsigned integer, big-endian)
   - `B (4 bytes)`: Batch size (unsigned integer)
   - `L (4 bytes)`: Sequence length (unsigned integer)
   - `d (4 bytes)`: Tensor representation dimension (unsigned integer)
   - `Scale Gamma (8 bytes)`: Dynamic symmetric quantization scale (float64 double)
   - `Bit-Width (4 bytes)`: Number of quantization bits (unsigned integer, e.g., 8)

2. **VTP Server Response (24 bytes)**:
   - `Magic Bytes (4 bytes)`: `VTPR`
   - `Loss (4 bytes)`: Current calculated total alignment loss (float32)
   - `MSE (4 bytes)`: Mean Squared Error component (float32)
   - `Cosine Distance (4 bytes)`: Cosine Distance component (float32)
   - `Orthogonal Penalty (4 bytes)`: Procrustes orthogonal regularization penalty (float32)
   - `Backprop Triggered (4 bytes)`: Flag indicating if inline micro-backprop was executed (1.0 = True, 0.0 = False)

---

### 🕹️ Multi-Process Execution Instructions

To run the distributed network mesh and real-time visualization, open **three separate terminals**:

#### 1️⃣ Terminal 1: Start the Executor Server
The server starts the Target Space listener ($d_2 = 512$), intercepts incoming quantized representations, performs dequantization, runs the projector, checks alignment entropy boundaries, and triggers the inline backprop:
```bash
python network/server.py
```

#### 2️⃣ Terminal 2: Run the Streamlit Dashboard
The dashboard uses scikit-learn PCA fitted to a stable target subspace to render a real-time, zero-jitter 3D visual convergence plot showing the two manifolds morphing and docking together:
```bash
streamlit run viz/dashboard.py
```

#### 3️⃣ Terminal 3: Launch the Analyst Client
The client connects to the server, generates dynamic latent states, quantizes them, packages the byte stream, and pushes them to the server while outputting the received metric responses:
```bash
python network/client.py
```

---

## 📄 License
This framework is licensed under the MIT License.
