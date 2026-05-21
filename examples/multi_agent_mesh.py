import torch
import torch.nn as nn
import sys
import os

# Inject parent directory into path for modular import compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runtime import LatentProjector, SynapseStream, EntropyRuntime

def main() -> None:
    # Fix random seed for exact performance evaluation
    torch.manual_seed(42)
    
    print("=" * 80)
    print("         SynapseLML (Latent Modeling Language) Multi-Agent Mesh")
    print("=" * 80)
    print("Scenario: 'Analyst Agent' communicating complex semantic representations")
    print("          directly into the latent space of an 'Executor Agent'.")
    print("-" * 80)
    print(f"Source Dimension (Analyst Agent)  : d1 = 256")
    print(f"Target Dimension (Executor Agent) : d2 = 512")
    print("Communication Protocol            : Vector Transport Protocol (VTP) - 8-Bit Quantized")
    print("Runtime Optimization              : Entropy-Based Runtime (EBR) inline feedback")
    print("=" * 80)
    
    # Initialize modular components
    projector = LatentProjector(d_source=256, d_target=512)
    stream = SynapseStream(bit_width=8)
    
    # Instantiate the EntropyRuntime with strict alignment constraints
    # threshold = 0.01 forces micro-backpropagation until high-fidelity alignment is met.
    runtime = EntropyRuntime(
        projector=projector, 
        lr=0.005, 
        threshold=0.01, 
        alpha=0.6,    # 60% weight on spatial distance (MSE)
        beta=0.02     # Orthogonality regularization coefficient
    )
    
    # Setup dimension dimensions for 3D hidden states [Batch, Sequence, Dim]
    batch_size = 4
    seq_len = 16
    
    # Instantiate static ground truth conceptual projection representing agent representation mapping
    true_mapping = torch.empty(256, 512)
    nn.init.orthogonal_(true_mapping)
    true_bias = torch.randn(512) * 0.02
    
    print(f"{'Epoch':<6} | {'Loss':<10} | {'MSE':<10} | {'Cos Dist':<10} | {'Orth Pen':<10} | {'EBR Action':<10}")
    print("-" * 80)
    
    for epoch in range(1, 21):
        # 1. Analyst Agent generates structured semantic latent states (with sequence dependencies)
        temporal_bias = torch.sin(torch.tensor(epoch * 0.4)).item()
        analyst_states = torch.randn(batch_size, seq_len, 256) * 0.5 + temporal_bias
        
        # 2. Target representation space expected by the Executor Agent (Ground Truth)
        executor_expected = torch.matmul(analyst_states, true_mapping) + true_bias
        
        # 3. Serialize and Quantize representation states via Vector Transport Protocol (VTP)
        packed_packet = stream.pack(analyst_states)
        unpacked_states = stream.unpack(packed_packet)
        
        # 4. Stream representation states through the Entropy-Based Runtime (EBR)
        projected_states, metrics = runtime.process(unpacked_states, executor_expected)
        
        action_taken = "MICRO_BP" if metrics["micro_backprop_triggered"] > 0 else "PASSIVE"
        
        print(
            f"{epoch:<6} | "
            f"{metrics['loss']:<10.6f} | "
            f"{metrics['mse']:<10.6f} | "
            f"{metrics['cosine_distance']:<10.6f} | "
            f"{metrics['orthogonal_penalty']:<10.6f} | "
            f"{action_taken:<10}"
        )
        
    print("-" * 80)
    
    # Validate final orthogonality deviation of the projection weight tensor
    final_orth_penalty = projector.get_orthogonal_penalty().item()
    print(f"Final Orthogonal Penalty (Frobenius deviation): {final_orth_penalty:.8f}")
    if final_orth_penalty < 1.0:
        print("STATUS: SUCCESS. Projector maintained absolute structure, preventing representation collapse.")
    else:
        print("STATUS: WARNING. Spatial distortion above recommended threshold.")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
