"""
Simple two-process GPU memory transfer test using NIXL + UCX.

Usage overview:
    # Terminal 1 (source, listens for sink connection)
    python tools/nixl_ucx_transfer_test.py --role source --listen --port 4567 \
        --device-id 0

    # Terminal 2 (sink, connects to source and pulls data into its GPU buffer)
    python tools/nixl_ucx_transfer_test.py --role sink --peer-host <source-ip> \
        --port 4567 --device-id 1

Key parameters:
    --num-blocks: number of blocks in the transfer (default: 10000)
    --block-len: size of each block in bytes (default: 32768 == 32KB)

The sink initiates a READ transfer from the source GPU into its own GPU buffer.
After completion, telemetry and basic data validation results are printed.
"""

import argparse
import json
import os
import socket
import struct
import time
import uuid
from typing import Any, Dict

import numpy as np
import torch

try:
    from nixl._api import nixl_agent, nixl_agent_config
    from nixl._bindings import nixlXferTelemetry
except Exception as exc:  # pragma: no cover - import guard for environments without nixl
    raise SystemExit(
        "Failed to import nixl. Make sure nixl is installed in the environment"
    ) from exc


DEFAULT_BLOCK_LEN = 32 * 1024
DEFAULT_NUM_BLOCKS = 10_000


# ----------------------------- networking utils -----------------------------

def _send_json(sock: socket.socket, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode()
    sock.sendall(struct.pack("!I", len(raw)))
    sock.sendall(raw)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def _recv_json(sock: socket.socket) -> Dict[str, Any]:
    size_raw = _recv_exact(sock, 4)
    (size,) = struct.unpack("!I", size_raw)
    payload = _recv_exact(sock, size)
    return json.loads(payload.decode())


def _establish_connection(peer_host: str, port: int, listen: bool) -> socket.socket:
    if listen:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((peer_host, port))
        server.listen(1)
        conn, _ = server.accept()
        server.close()
        return conn

    return socket.create_connection((peer_host, port))


# ------------------------------ nixl utilities ------------------------------

def _init_nixl_agent(backends: list[str]) -> Any:
    os.environ.setdefault("NIXL_TELEMETRY_ENABLE", "1")
    config = nixl_agent_config(backends=backends) if nixl_agent_config else None
    return nixl_agent(str(uuid.uuid4()), config)


def _telemetry_to_dict(telemetry: Any) -> Dict[str, Any]:
    if telemetry is None:
        return {}
    if isinstance(telemetry, dict):
        return telemetry
    if isinstance(telemetry, nixlXferTelemetry):
        # nixlXferTelemetry exposes public attributes.
        attrs: Dict[str, Any] = {}
        for key in dir(telemetry):
            if key.startswith("_"):
                continue
            value = getattr(telemetry, key)
            if callable(value):
                continue
            attrs[key] = value
        return attrs
    try:
        return dict(telemetry)
    except Exception:
        return {"telemetry": str(telemetry)}


# ------------------------------- main routine -------------------------------

def _prep_memory(
    agent: Any,
    tensor: torch.Tensor,
    memory_type: str,
    num_blocks: int,
    block_len: int,
    backends: list[str],
) -> tuple[list[int], int]:
    base_addr = tensor.data_ptr()
    total_bytes = tensor.numel()
    device_id = tensor.device.index or 0

    caches_data = [(base_addr, total_bytes, device_id, "")]
    reg_descs = agent.get_reg_descs(caches_data, memory_type)
    agent.register_memory(reg_descs, backends=backends)

    blocks_data = []
    for block_id in range(num_blocks):
        offset = block_id * block_len
        blocks_data.append((base_addr + offset, block_len, device_id))

    xfer_descs = agent.get_xfer_descs(blocks_data, memory_type)
    local_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_descs)
    return list(range(num_blocks)), local_handle


def _build_remote_descs(agent: Any, remote_meta: Dict[str, Any], memory_type: str, agent_name: str) -> int:
    base_addr = int(remote_meta["base_addr"])
    num_blocks = int(remote_meta["num_blocks"])
    block_len = int(remote_meta["block_len"])
    device_id = int(remote_meta["device_id"])

    blocks_data = []
    for block_id in range(num_blocks):
        offset = block_id * block_len
        blocks_data.append((base_addr + offset, block_len, device_id))

    descs = agent.get_xfer_descs(blocks_data, memory_type)
    return agent.prep_xfer_dlist(agent_name, descs)


def _build_metadata(agent: Any, tensor: torch.Tensor, num_blocks: int, block_len: int, memory_type: str) -> Dict[str, Any]:
    return {
        "agent_metadata": agent.get_agent_metadata().hex(),
        "base_addr": tensor.data_ptr(),
        "num_blocks": num_blocks,
        "block_len": block_len,
        "device_id": tensor.device.index or 0,
        "memory_type": memory_type,
    }


def _wait_for_notif(agent: Any, peer: str, timeout_s: float = 30.0) -> list[bytes]:
    deadline = time.time() + timeout_s
    messages: list[bytes] = []
    while time.time() < deadline:
        notifs = agent.get_new_notifs()
        peer_msgs = notifs.get(peer, [])
        if peer_msgs:
            messages.extend(peer_msgs)
            break
        time.sleep(0.05)
    return messages


def main(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device_id)
    device = torch.device(f"cuda:{args.device_id}")
    memory_type = "VRAM" if device.type == "cuda" else "DRAM"

    total_bytes = args.num_blocks * args.block_len
    tensor = torch.arange(total_bytes, dtype=torch.uint8, device=device)

    agent = _init_nixl_agent(args.backends)

    local_desc_ids, local_handle = _prep_memory(
        agent,
        tensor,
        memory_type,
        args.num_blocks,
        args.block_len,
        args.backends,
    )
  
    sock = _establish_connection(args.peer_host, args.port, args.listen)
    with sock:
        local_meta = _build_metadata(agent, tensor, args.num_blocks, args.block_len, memory_type)
        _send_json(sock, local_meta)
        remote_meta = _recv_json(sock)

        remote_agent_name = agent.add_remote_agent(bytes.fromhex(remote_meta["agent_metadata"]))

        remote_handle = _build_remote_descs(
            agent, remote_meta, remote_meta.get("memory_type", memory_type), remote_agent_name
        )
        remote_desc_ids = list(range(int(remote_meta["num_blocks"])))

        peer_block_len = int(remote_meta["block_len"])
        transfer_blocks = min(len(local_desc_ids), len(remote_desc_ids))
        if transfer_blocks < len(local_desc_ids) or transfer_blocks < len(remote_desc_ids):
            print(
                f"Clamping transfer to {transfer_blocks} blocks (local={len(local_desc_ids)},"
                f" remote={len(remote_desc_ids)})"
            )
        local_ids = np.array(local_desc_ids[:transfer_blocks])
        remote_ids = np.array(remote_desc_ids[:transfer_blocks])

        if args.role == "sink":
            start = time.perf_counter()
            handle = agent.make_prepped_xfer(
                "READ", local_handle, local_ids, remote_handle, remote_ids
            )
            agent.transfer(handle)

            while True:
                state = agent.check_xfer_state(handle)
                if state == "DONE":
                    break
                time.sleep(args.poll_interval)

            duration = time.perf_counter() - start
            telemetry = _telemetry_to_dict(agent.get_xfer_telemetry(handle))
            agent.release_xfer_handle(handle)
            agent.send_notif(remote_agent_name, b"transfer_complete")

            expected = torch.arange(total_bytes, dtype=torch.uint8, device=device)
            mismatch = int(torch.count_nonzero(tensor != expected).item())

            bytes_transferred = transfer_blocks * min(args.block_len, peer_block_len)

            print(json.dumps(
                {
                    "role": args.role,
                    "bytes": bytes_transferred,
                    "num_blocks": transfer_blocks,
                    "block_len": args.block_len,
                    "duration_s": duration,
                    "bandwidth_gbps": (bytes_transferred / duration) / 1e9,
                    "mismatched_bytes": mismatch,
                    "telemetry": telemetry,
                },
                indent=2,
            ))
        else:
            done = _wait_for_notif(agent, remote_agent_name)
            print(f"Source received notifications from {remote_agent_name}: {done}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIXL + UCX GPU transfer test")
    parser.add_argument("--role", choices=["source", "sink"], required=True, help="source holds data; sink pulls data")
    parser.add_argument("--listen", action="store_true", help="listen for a TCP connection instead of connecting")
    parser.add_argument("--peer-host", default="127.0.0.1", help="peer hostname or IP (ignored when --listen is set)")
    parser.add_argument("--port", type=int, default=4567, help="port for control channel")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id to use")
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS, help="number of blocks per transfer")
    parser.add_argument("--block-len", type=int, default=DEFAULT_BLOCK_LEN, help="size of each block in bytes")
    parser.add_argument("--backends", nargs="+", default=["UCX"], help="NIXL backends to enable")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="seconds between transfer state polls")

    args = parser.parse_args()
    main(args)
