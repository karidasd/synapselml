import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

class LatentProjector(nn.Module):
    """
    PyTorch module projecting hidden states from Source Manifold Space (d_source)
    to Target Manifold Space (d_target). Implements linear scaling with Procrustes orthogonal regularization.
    """
    def __init__(self, d_source: int = 256, d_target: int = 512) -> None:
        super().__init__()
        self.d_source = d_source
        self.d_target = d_target
        
        # Learnable translation weights and bias
        self.linear = nn.Linear(d_source, d_target, bias=True)
        
        # Initialize weights with orthogonal initialization
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass projecting source states.
        
        Args:
            x: Input tensor representing source latent states.
               Shape: [batch_size, seq_len, d_source]
               
        Returns:
            Projected target tensor.
            Shape: [batch_size, seq_len, d_target]
        """
        return self.linear(x)

    def get_orthogonal_penalty(self) -> torch.Tensor:
        """
        Computes the Frobenius norm penalty enforcing weight matrix orthogonality.
        For W of shape [d_target, d_source]:
        If d_source <= d_target, we check W^T * W - I_d_source.
        If d_source > d_target, we check W * W^T - I_d_target.
        
        Returns:
            Frobenius norm penalty scalar tensor.
        """
        w = self.linear.weight # Shape: [d_target, d_source]
        if self.d_source <= self.d_target:
            identity = torch.eye(self.d_source, device=w.device, dtype=w.dtype)
            prod = torch.matmul(w.t(), w)
        else:
            identity = torch.eye(self.d_target, device=w.device, dtype=w.dtype)
            prod = torch.matmul(w, w.t())
        return torch.norm(prod - identity, p='fro') ** 2


class SynapseStream:
    """
    Simulates high-dimensional tensor serialization, packaging, and dynamic 
    symmetric quantization/dequantization over the Vector Transport Protocol (VTP).
    """
    def __init__(self, bit_width: int = 8) -> None:
        self.bit_width = bit_width

    def pack(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Quantizes a float32 tensor into a packed int8 payload with metadata.
        
        Args:
            tensor: High-dimensional float tensor.
                    Shape: [batch_size, seq_len, d_model]
                    
        Returns:
            Dictionary containing quantized payload, scale factor, and original shape.
        """
        # Calculate dynamic range scale factor
        max_val = torch.max(torch.abs(tensor))
        scale = 1.0 if max_val == 0 else (2**(self.bit_width - 1) - 1) / max_val.item()
        
        # Symmetric quantization mapping and clipping
        quantized = torch.clamp(
            torch.round(tensor * scale),
            -(2**(self.bit_width - 1)),
            2**(self.bit_width - 1) - 1
        ).to(torch.int8)
        
        return {
            "payload": quantized,
            "scale": scale,
            "shape": tensor.shape
        }

    def unpack(self, packet: Dict[str, Any]) -> torch.Tensor:
        """
        Dequantizes a packet structure back into float32 representation.
        
        Args:
            packet: Packet structure containing quantized payload and metadata.
            
        Returns:
            Dequantized float32 tensor.
            Shape: Same as the original input shape.
        """
        payload = packet["payload"]
        scale = packet["scale"]
        
        # Dequantization mapping
        return payload.to(torch.float32) / scale


class EntropyRuntime:
    """
    Real-time execution engine tracking representation alignment loss.
    Triggers inline micro-backpropagation steps to optimize projection mapping
    upon entropy threshold violations.
    """
    def __init__(
        self, 
        projector: LatentProjector, 
        lr: float = 0.01, 
        threshold: float = 0.05, 
        alpha: float = 0.5, 
        beta: float = 0.1
    ) -> None:
        self.projector = projector
        self.optimizer = torch.optim.Adam(self.projector.parameters(), lr=lr)
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def compute_loss(self, projected: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the joint loss: alpha * MSE + (1 - alpha) * Cosine Distance + beta * Orthogonal Penalty.
        
        Args:
            projected: Projected states [batch_size, seq_len, d_target]
            target: Target expected states [batch_size, seq_len, d_target]
            
        Returns:
            Tuple of tensors (total_loss, mse_loss, cosine_distance_loss, orth_penalty).
        """
        # Coordinate-wise alignment check
        mse_loss = nn.functional.mse_loss(projected, target)
        
        # Directional alignment check (semantic similarity)
        p_flat = projected.view(-1, projected.shape[-1])
        t_flat = target.view(-1, target.shape[-1])
        cos_similarity = nn.functional.cosine_similarity(p_flat, t_flat, dim=-1)
        cos_distance = 1.0 - torch.mean(cos_similarity)
        
        # Structural distortion alignment check
        orth_penalty = self.projector.get_orthogonal_penalty()
        
        # Joint alignment loss optimization objective
        total_loss = (self.alpha * mse_loss) + ((1.0 - self.alpha) * cos_distance) + (self.beta * orth_penalty)
        
        return total_loss, mse_loss, cos_distance, orth_penalty

    def process(self, source_states: torch.Tensor, target_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Processes a communication cycle, measuring loss and executing inline correction if needed.
        
        Args:
            source_states: Tensor of source representation states.
                           Shape: [batch_size, seq_len, d_source]
            target_states: Tensor of target ground truth representation states.
                           Shape: [batch_size, seq_len, d_target]
                           
        Returns:
            Tuple containing the final projected states and execution metrics dictionary.
        """
        # 1. Direct Linear Projection mapping
        projected = self.projector(source_states)
        
        # 2. Joint Loss evaluation
        loss, mse, cos, orth = self.compute_loss(projected, target_states)
        
        metrics = {
            "loss": loss.item(),
            "mse": mse.item(),
            "cosine_distance": cos.item(),
            "orthogonal_penalty": orth.item(),
            "micro_backprop_triggered": 0.0
        }
        
        # 3. Dynamic Threshold check and correction
        if loss.item() > self.threshold:
            metrics["micro_backprop_triggered"] = 1.0
            self.micro_backprop(source_states, target_states)
            
            # Post-backpropagation re-evaluation
            projected = self.projector(source_states)
            loss, mse, cos, orth = self.compute_loss(projected, target_states)
            metrics["loss"] = loss.item()
            metrics["mse"] = mse.item()
            metrics["cosine_distance"] = cos.item()
            metrics["orthogonal_penalty"] = orth.item()
            
        return projected, metrics

    def micro_backprop(self, source_states: torch.Tensor, target_states: torch.Tensor) -> None:
        """
        Executes a real-time localized backward pass to update projector parameters.
        
        Args:
            source_states: Tensor of source states.
            target_states: Tensor of target states.
        """
        self.optimizer.zero_grad()
        projected = self.projector(source_states)
        loss, _, _, _ = self.compute_loss(projected, target_states)
        loss.backward()
        self.optimizer.step()
