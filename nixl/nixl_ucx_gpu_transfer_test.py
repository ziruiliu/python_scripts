# SPDX-License-Identifier: Apache-2.0
"""
A lightweight two-process test tool for exercising NIXL + UCX transfers.

The script allocates GPU memory on both sides, registers it with the NIXL
agent, exchanges metadata over ZeroMQ, and issues a write transfer from the
sender to the receiver. Wall-clock duration and throughput are printed after
the transfer finishes, and the receiver verifies the transferred payload.

Example usage (two shells):

Receiver:
    python nixl_ucx_gpu_transfer_test.py --role receiver --listen-port 5555

Sender (in another shell on the same host):
    python nixl_ucx_gpu_transfer_test.py --role sender --peer localhost:5555

Important notes:
* Both processes must select distinct GPUs (``--device``) when running on a
  multi-GPU host.
* Defaults send 18 objects of 36MB each using the UCX backend.
* The script is intentionally minimal and does not depend on LMCache storage
  abstractions or KV cache layout.
"""

from __future__ import annotations

# Standard
import argparse
import time
import uuid
from typing import Any, List, Sequence

# Third Party
import msgspec
import torch
import zmq


def _require_nixl():
    try:
        from nixl._api import nixl_agent as NixlAgent
        from nixl._api import nixl_agent_config
    except ImportError as err:  # pragma: no cover - import guard
        raise RuntimeError(
            "NIXL is required for this test. Install nixl and ensure the UCX "
            "backend is available."
        ) from err

    return NixlAgent, nixl_agent_config


class InitPacket(msgspec.Struct):
    meta_bytes: bytes
    serialized_xfer_descs: bytes
    object_count: int
    object_bytes: int


class CompletionPacket(msgspec.Struct):
    duration_s: float
    throughput_gbps: float
    bytes_moved: int


def build_agent(backends: Sequence[str]):
    NixlAgent, nixl_agent_config = _require_nixl()
    agent_name = str(uuid.uuid4())
    agent = NixlAgent(agent_name, nixl_agent_config(backends=list(backends)))
    return agent, agent_name


def allocate_buffer(device: str, object_bytes: int, object_count: int) -> torch.Tensor:
    total_bytes = object_bytes * object_count
    torch.cuda.set_device(device)
    buffer = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    return buffer


def prep_local_descriptors(
    agent,
    buffer: torch.Tensor,
    object_bytes: int,
    object_count: int,
) -> tuple[List[tuple[int, int, int, str]], Any, Any, List[int]]:
    device_index = buffer.device.index
    buffer_ptr = buffer.data_ptr()
    total_bytes = object_bytes * object_count

    reg_list = [(buffer_ptr, total_bytes, device_index, "gpu-buffer")]
    reg_descs = agent.get_reg_descs(reg_list, mem_type="cuda")
    agent.register_memory(reg_descs)

    xfer_desc = []
    indices = []
    for i in range(object_count):
        offset = buffer_ptr + i * object_bytes
        xfer_desc.append((offset, object_bytes, device_index))
        indices.append(i)

    xfer_descs = agent.get_xfer_descs(xfer_desc, mem_type="cuda")
    xfer_handler = agent.prep_xfer_dlist("", xfer_descs, mem_type="cuda")
    serialized_descs = agent.get_serialized_descs(xfer_descs)

    return reg_descs, xfer_handler, serialized_descs, indices


def exchange_metadata(
    context: zmq.Context,
    role: str,
    listen_port: int | None,
    peer: str | None,
    local_packet: InitPacket,
):
    socket = None
    if role == "receiver":
        assert listen_port is not None
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{listen_port}")
        request = socket.recv()
        remote_packet = msgspec.msgpack.decode(request, type=InitPacket)
        socket.send(msgspec.msgpack.encode(local_packet))
    else:
        assert peer is not None
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{peer}")
        socket.send(msgspec.msgpack.encode(local_packet))
        response = socket.recv()
        remote_packet = msgspec.msgpack.decode(response, type=InitPacket)
    return socket, remote_packet


def verify_received_buffer(
    buffer: torch.Tensor,
    expected_byte: int,
) -> tuple[bool, int | None]:
    """Check that the received buffer matches the expected byte pattern."""

    expected = torch.full_like(buffer, fill_value=expected_byte)
    matches = torch.equal(buffer, expected)
    mismatch_index = None
    if not matches:
        # Report the first byte index that differs for easier debugging.
        diff_indices = (buffer != expected).nonzero(as_tuple=False)
        if diff_indices.numel() > 0:
            mismatch_index = diff_indices[0].item()

    return matches, mismatch_index


def compute_throughput(bytes_moved: int, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return (bytes_moved / duration) / (1024**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--peer", help="peer host:port (for sender)")
    parser.add_argument("--listen-port", type=int, help="port to bind for receiver")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use")
    parser.add_argument("--object-count", type=int, default=18)
    parser.add_argument("--object-mb", type=int, default=36)
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size used when building transfer descriptors.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="NIXL backend (default: UCX). May be passed multiple times.",
    )

    args = parser.parse_args()
    backends = args.backend or ["UCX"]
    object_bytes = args.object_mb * 1024 * 1024
    if object_bytes % args.page_size != 0:
        raise ValueError("object size must be a multiple of --page-size")
    buffer = allocate_buffer(args.device, object_bytes, args.object_count)

    if args.role == "sender":
        # Seed sender buffer with a deterministic byte pattern for validation.
        buffer.fill_(ord("A"))
    else:
        buffer.zero_()

    agent, agent_name = build_agent(backends)
    reg_descs, xfer_handler, serialized_descs, indices = prep_local_descriptors(
        agent,
        buffer,
        object_bytes,
        args.object_count,
    )

    context = zmq.Context()
    init_packet = InitPacket(
        meta_bytes=agent.get_agent_metadata(),
        serialized_xfer_descs=serialized_descs,
        object_count=args.object_count,
        object_bytes=object_bytes,
    )

    socket, remote_packet = exchange_metadata(
        context,
        args.role,
        args.listen_port,
        args.peer,
        init_packet,
    )

    remote_agent_name = agent.add_remote_agent(remote_packet.meta_bytes)
    remote_xfer_descs = agent.deserialize_descs(remote_packet.serialized_xfer_descs)
    remote_xfer_handler = agent.prep_xfer_dlist(
        remote_agent_name, remote_xfer_descs, mem_type="cuda"
    )

    if args.role == "receiver":
        # Wait for the sender to declare completion, then validate the buffer.
        message = socket.recv()
        completion = msgspec.msgpack.decode(message, type=CompletionPacket)

        torch.cuda.synchronize()
        ok, mismatch_index = verify_received_buffer(buffer, ord("A"))

        socket.send(
            msgspec.msgpack.encode({"verification_ok": ok, "mismatch_index": mismatch_index})
        )

        print("\n=== Transfer complete (receiver view) ===")
        print(f"objects: {args.object_count}, size per object: {args.object_mb} MB")
        print(f"duration_s: {completion.duration_s:.4f}")
        print(f"throughput_gbps: {completion.throughput_gbps:.2f}")
        print(f"bytes_moved: {completion.bytes_moved}")
        print(f"verification_ok: {ok}")
        if not ok:
            print(f"first mismatching byte index: {mismatch_index}")

        agent.release_dlist_handle(remote_xfer_handler)
        agent.release_dlist_handle(xfer_handler)
        agent.deregister_memory(reg_descs)
        return

    # Sender: build and launch the transfer.
    bytes_moved = object_bytes * args.object_count
    torch.cuda.synchronize()
    start = time.perf_counter()
    handle = agent.make_prepped_xfer(
        "WRITE", xfer_handler, indices, remote_xfer_handler, indices
    )

    agent.transfer(handle)
    while True:
        status = agent.check_xfer_state(handle)
        if status == "DONE":
            break
        if status == "ERR":
            raise RuntimeError("NIXL transfer reported an error")
        time.sleep(0.001)

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    agent.release_xfer_handle(handle)

    if hasattr(agent, "send_notif"):
        try:
            agent.send_notif(remote_agent_name, b"transfer-complete")
        except Exception:
            # Notification is best-effort for demonstration purposes.
            pass

    throughput_gbps = compute_throughput(bytes_moved, duration)

    completion = CompletionPacket(
        duration_s=duration,
        throughput_gbps=throughput_gbps,
        bytes_moved=bytes_moved,
    )
    socket.send(msgspec.msgpack.encode(completion))
    ack = socket.recv()
    ack_info = msgspec.msgpack.decode(ack)

    print("\n=== Transfer complete (sender view) ===")
    print(f"objects: {args.object_count}, size per object: {args.object_mb} MB")
    print(f"duration_s: {duration:.4f}")
    print(f"throughput_gbps: {throughput_gbps:.2f}")
    print(f"verification_ok (receiver): {ack_info.get('verification_ok')}")
    mismatch_index = ack_info.get("mismatch_index")
    if mismatch_index is not None:
        print(f"first mismatching byte index: {mismatch_index}")

    agent.release_dlist_handle(remote_xfer_handler)
    agent.release_dlist_handle(xfer_handler)
    agent.deregister_memory(reg_descs)


if __name__ == "__main__":
    main()
