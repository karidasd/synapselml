import asyncio
import os
import sys
import struct
import pickle
import tempfile
import argparse
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn

# Inject project root and its parent into path for robust modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
project_parent = os.path.abspath(os.path.join(project_root, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

from core.runtime import LatentProjector, SynapseStream, EntropyRuntime

HEADER_FORMAT = ">4sIIII d I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
TELEMETRY_FILE = os.getenv("TELEMETRY_PATH", "synapselml_telemetry.pkl")

def write_telemetry(data: Dict[str, Any], path: str = TELEMETRY_FILE) -> None:
    """Writes telemetry data atomically to prevent race conditions during dashboard reads."""
    tmp_dir = os.path.dirname(path) or "."
    if tmp_dir and tmp_dir != ".":
        os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=tmp_dir, delete=False) as f:
        pickle.dump(data, f)
        tmp_name = f.name
    try:
        os.replace(tmp_name, path)
    except Exception as e:

        print(f"Telemetry write warning: {e}")
        try:
            os.remove(tmp_name)
        except Exception:
            pass

def deserialize_header(header_bytes: bytes) -> Tuple[bytes, int, int, int, int, float, int]:
    """Unpacks the 32-byte VTP binary header."""
    magic, payload_size, B, L, d, scale, bit_width = struct.unpack(HEADER_FORMAT, header_bytes)
    return magic, payload_size, B, L, d, scale, bit_width

async def main() -> None:
    parser = argparse.ArgumentParser(description="SynapseLML Executor Agent TCP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch execution device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("=" * 80)
    print("         SynapseLML (Latent Modeling Language) Executor Server")
    print("=" * 80)
    print(f"Target Manifold Dimension : d2 = 512")
    print(f"Network Listening on     : {args.host}:{args.port}")
    print(f"Compute Device            : {device}")
    print("=" * 80)

    # Initialize a fixed ground truth projection target.
    # The Analyst client generates states, and the server aligns them to this space.
    torch.manual_seed(42)
    true_mapping = torch.empty(256, 512)
    nn.init.orthogonal_(true_mapping)
    true_bias = torch.randn(512) * 0.02
    
    true_mapping = true_mapping.to(device)
    true_bias = true_bias.to(device)

    # SynapseStream instance for unpacking FP8 payloads
    stream = SynapseStream(bit_width=8)

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        print(f"\n[+] Analyst Agent connected from {peer[0]}:{peer[1]}")
        
        # Instantiate a fresh projection model and runtime optimizer for this session.
        # This demonstrates convergence starting from high alignment loss.
        projector = LatentProjector(d_source=256, d_target=512).to(device)
        runtime = EntropyRuntime(
            projector=projector,
            lr=0.005,
            threshold=0.01,
            alpha=0.6,
            beta=0.02
        )
        
        # Clear any stale telemetry
        if os.path.exists(TELEMETRY_FILE):
            try:
                os.remove(TELEMETRY_FILE)
            except Exception:
                pass
                
        step_count = 0
        history: List[Dict[str, float]] = []

        try:
            while True:
                # Read 32-byte header
                header_bytes = await reader.readexactly(HEADER_SIZE)
                magic, payload_size, B, L, d, scale, bit_width = deserialize_header(header_bytes)
                
                if magic != b"VTP\x00":
                    print(f"[-] Protocol error: Invalid magic header {magic}")
                    break
                
                # Read payload bytes
                payload_bytes = await reader.readexactly(payload_size)
                
                # Reconstruct and unpack tensor
                quantized_np = np.frombuffer(payload_bytes, dtype=np.int8).copy().reshape(B, L, d)
                quantized_tensor = torch.from_numpy(quantized_np).to(device)
                
                packet = {
                    "payload": quantized_tensor,
                    "scale": scale,
                    "shape": (B, L, d)
                }
                
                # Dequantize back to float32
                unpacked_states = stream.unpack(packet).to(device)
                
                step_count += 1
                
                # Compute ground truth targets using the true target mapping
                executor_expected = torch.matmul(unpacked_states, true_mapping) + true_bias
                
                # Process via EntropyRuntime: checks alignment, runs micro-backprop if loss > threshold
                projected_states, metrics = runtime.process(unpacked_states, executor_expected)
                
                action_taken = "MICRO_BP" if metrics["micro_backprop_triggered"] > 0 else "PASSIVE"
                print(
                    f"Epoch {step_count:02d} | Loss: {metrics['loss']:.6f} | "
                    f"MSE: {metrics['mse']:.6f} | Cos Dist: {metrics['cosine_distance']:.6f} | "
                    f"Orth Pen: {metrics['orthogonal_penalty']:.6f} | Action: {action_taken}"
                )
                
                # Append metrics to historical list for logging and graphing
                history.append({
                    "epoch": step_count,
                    "loss": metrics["loss"],
                    "mse": metrics["mse"],
                    "cosine_distance": metrics["cosine_distance"],
                    "orthogonal_penalty": metrics["orthogonal_penalty"],
                    "micro_backprop_triggered": metrics["micro_backprop_triggered"]
                })
                
                # Write state telemetry for dashboard visualization
                telemetry_data = {
                    "epoch": step_count,
                    "source_states": unpacked_states.detach().cpu().numpy(),
                    "projected_states": projected_states.detach().cpu().numpy(),
                    "target_states": executor_expected.detach().cpu().numpy(),
                    "metrics": metrics,
                    "history": history
                }
                write_telemetry(telemetry_data)
                
                # Send VTP Response Packet (24 bytes) back to client
                response_bytes = struct.pack(
                    ">4sfffff",
                    b"VTPR",
                    metrics["loss"],
                    metrics["mse"],
                    metrics["cosine_distance"],
                    metrics["orthogonal_penalty"],
                    metrics["micro_backprop_triggered"]
                )
                writer.write(response_bytes)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            print(f"[-] Analyst Agent disconnected from {peer[0]}:{peer[1]}")
        except Exception as e:
            print(f"[-] Connection handler error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, args.host, args.port)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Executor Server terminated by user.")
        sys.exit(0)
