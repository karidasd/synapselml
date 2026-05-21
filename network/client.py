import asyncio
import os
import sys
import struct
import argparse
import time
from typing import Tuple
import torch

# Inject project root and its parent into path for robust modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
project_parent = os.path.abspath(os.path.join(project_root, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

from core.runtime import SynapseStream

HEADER_FORMAT = ">4sIIII d I"
RESPONSE_FORMAT = ">4sfffff"
RESPONSE_SIZE = struct.calcsize(RESPONSE_FORMAT)

def serialize_packet(tensor: torch.Tensor, scale: float, bit_width: int) -> bytes:
    """Packs the header and quantized payload into VTP protocol bytes."""
    B, L, d = tensor.shape
    quantized_payload = tensor.cpu().numpy().astype("int8").tobytes()
    payload_size = len(quantized_payload)
    
    header = struct.pack(
        HEADER_FORMAT,
        b"VTP\x00",
        payload_size,
        B,
        L,
        d,
        scale,
        bit_width
    )
    return header + quantized_payload

def deserialize_response(response_bytes: bytes) -> Tuple[bytes, float, float, float, float, float]:
    """Unpacks the server feedback response packet."""
    magic, loss, mse, cos_dist, orth_pen, triggered = struct.unpack(RESPONSE_FORMAT, response_bytes)
    return magic, loss, mse, cos_dist, orth_pen, triggered

async def stream_data() -> None:
    parser = argparse.ArgumentParser(description="SynapseLML Analyst Agent TCP Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs to stream")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between epochs to visualize morphing")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size of hidden states")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length of hidden states")
    args = parser.parse_args()

    print("=" * 80)
    print("         SynapseLML (Latent Modeling Language) Analyst Client")
    print("=" * 80)
    print(f"Source Manifold Dimension : d1 = 256")
    print(f"Connecting to Server      : {args.host}:{args.port}")
    print(f"Streaming Configuration   : {args.epochs} epochs, {args.delay}s delay per epoch")
    print("=" * 80)

    # Initialize client-side quantization stream
    stream = SynapseStream(bit_width=8)
    
    # Fix random seed on client for exact simulation reproduction
    torch.manual_seed(42)

    try:
        reader, writer = await asyncio.open_connection(args.host, args.port)
        print("[+] Established connection to Executor Server. Streaming sequence...")
        print("-" * 80)
        print(f"{'Epoch':<6} | {'Loss':<10} | {'MSE':<10} | {'Cos Dist':<10} | {'Orth Pen':<10} | {'EBR Action':<10}")
        print("-" * 80)

        for epoch in range(1, args.epochs + 1):
            # 1. Analyst Agent generates structured semantic latent states (with sequence dependencies)
            temporal_bias = torch.sin(torch.tensor(epoch * 0.4)).item()
            analyst_states = torch.randn(args.batch_size, args.seq_len, 256) * 0.5 + temporal_bias
            
            # 2. Serialize and Quantize representation states via Vector Transport Protocol (VTP)
            packed_packet = stream.pack(analyst_states)
            payload_bytes = serialize_packet(
                tensor=packed_packet["payload"],
                scale=packed_packet["scale"],
                bit_width=8
            )
            
            # 3. Stream payload bytes over the TCP socket
            writer.write(payload_bytes)
            await writer.drain()
            
            # 4. Await response feedback containing alignment metrics
            response_bytes = await reader.readexactly(RESPONSE_SIZE)
            magic, loss, mse, cos_dist, orth_pen, triggered = deserialize_response(response_bytes)
            
            if magic != b"VTPR":
                print(f"\n[-] Received invalid response magic: {magic}")
                break
                
            action_taken = "MICRO_BP" if triggered > 0 else "PASSIVE"
            print(
                f"{epoch:<6} | "
                f"{loss:<10.6f} | "
                f"{mse:<10.6f} | "
                f"{cos_dist:<10.6f} | "
                f"{orth_pen:<10.6f} | "
                f"{action_taken:<10}"
            )
            
            # Pause between epochs to allow visualization software to capture and render smoothly
            await asyncio.sleep(args.delay)
            
        print("-" * 80)
        print("[+] Finished streaming representation dataset. Connection closed.")
        writer.close()
        await writer.wait_closed()
        
    except ConnectionRefusedError:
        print("[-] Connection refused. Ensure the Executor Server is running.")
    except Exception as e:
        print(f"[-] Client runtime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(stream_data())
    except KeyboardInterrupt:
        print("\n[!] Analyst Client terminated by user.")
        sys.exit(0)
