# Copyright (c) 2024, Adaptive Pipeline Parallelism Research
# Bandwidth profiling for network topology characterization

import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from .topology import NetworkTopology


def measure_p2p_bandwidth(
    src_rank: int,
    dst_rank: int,
    tensor_size_mb: int = 100,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> Tuple[Optional[float], Optional[float]]:
    """Measure point-to-point bandwidth between two ranks.

    Args:
        src_rank: Source rank
        dst_rank: Destination rank
        tensor_size_mb: Size of tensor to transfer in MB
        num_iterations: Number of iterations to average
        warmup_iterations: Warmup iterations (not counted)

    Returns:
        (bandwidth_gbps, latency_ms): Bandwidth in GB/s and latency in ms,
                                       or (None, None) for non-participating ranks
    """
    rank = dist.get_rank()

    # Create tensor (float32 = 4 bytes per element)
    tensor_size = tensor_size_mb * 1024 * 1024 // 4
    tensor = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    # Warmup
    if rank == src_rank:
        for _ in range(warmup_iterations):
            dist.send(tensor, dst=dst_rank)
    elif rank == dst_rank:
        for _ in range(warmup_iterations):
            dist.recv(tensor, src=src_rank)

    # Synchronize before measurement
    dist.barrier()

    # Actual measurement
    if rank == src_rank:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            dist.send(tensor, dst=dst_rank)
        end.record()

        # Wait for completion
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)

        # Calculate bandwidth
        bytes_transferred = tensor_size * 4 * num_iterations  # 4 bytes per float32
        bandwidth_gbps = (bytes_transferred / (elapsed_ms / 1000)) / 1e9
        latency_ms = elapsed_ms / num_iterations

        return bandwidth_gbps, latency_ms

    elif rank == dst_rank:
        for _ in range(num_iterations):
            dist.recv(tensor, src=src_rank)
        return None, None  # Receiver doesn't compute

    else:
        # Other ranks wait
        return None, None


def measure_bidirectional_bandwidth(
    rank_a: int,
    rank_b: int,
    tensor_size_mb: int = 100,
    num_iterations: int = 10
) -> Tuple[Optional[float], Optional[float]]:
    """Measure bidirectional bandwidth (both directions simultaneously).

    This tests the full-duplex capability of the network.

    Args:
        rank_a: First rank
        rank_b: Second rank
        tensor_size_mb: Size of tensor to transfer in MB
        num_iterations: Number of iterations

    Returns:
        (bandwidth_gbps, latency_ms): Combined bandwidth for both directions
    """
    rank = dist.get_rank()
    tensor_size = tensor_size_mb * 1024 * 1024 // 4
    tensor = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    dist.barrier()

    if rank == rank_a or rank == rank_b:
        other_rank = rank_b if rank == rank_a else rank_a

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            # Simultaneous send and receive
            send_req = dist.isend(tensor, dst=other_rank)
            recv_req = dist.irecv(tensor, src=other_rank)
            send_req.wait()
            recv_req.wait()
        end.record()

        end.synchronize()
        elapsed_ms = start.elapsed_time(end)

        # Bandwidth is for both directions
        bytes_transferred = tensor_size * 4 * num_iterations * 2  # Both directions
        bandwidth_gbps = (bytes_transferred / (elapsed_ms / 1000)) / 1e9

        return bandwidth_gbps, elapsed_ms / num_iterations

    return None, None


def profile_intra_node_bandwidth(topology: NetworkTopology) -> Dict[int, float]:
    """Profile bandwidth within each node.

    Args:
        topology: NetworkTopology object with node information

    Returns:
        Dictionary mapping node_id to intra-node bandwidth (GB/s)
    """
    rank = dist.get_rank()
    results = {}

    for node_id, node_info in topology.nodes.items():
        if len(node_info.gpu_ranks) < 2:
            # Single GPU on node, no intra-node communication
            results[node_id] = float('inf')  # Conceptually infinite (no network)
            continue

        # Use first two GPUs on the node as representatives
        rank_a = node_info.gpu_ranks[0]
        rank_b = node_info.gpu_ranks[1]

        bw, _ = measure_p2p_bandwidth(rank_a, rank_b, tensor_size_mb=100)

        if rank == rank_a and bw is not None:
            results[node_id] = bw

    # Gather results to all ranks
    all_results = [None] * dist.get_world_size()
    dist.all_gather_object(all_results, results if results else {})

    # Merge results
    merged = {}
    for r in all_results:
        if r:
            merged.update(r)

    return merged


def profile_inter_node_bandwidth(topology: NetworkTopology) -> Dict[Tuple[int, int], float]:
    """Profile bandwidth between nodes.

    Args:
        topology: NetworkTopology object with node information

    Returns:
        Dictionary mapping (src_node, dst_node) to bandwidth (GB/s)
    """
    rank = dist.get_rank()
    results = {}

    # Get list of node pairs to measure
    node_ids = sorted(topology.nodes.keys())

    for i, src_node in enumerate(node_ids):
        for dst_node in node_ids[i+1:]:  # Only upper triangle (assume symmetric)
            # Use first GPU from each node
            src_rank = topology.nodes[src_node].gpu_ranks[0]
            dst_rank = topology.nodes[dst_node].gpu_ranks[0]

            bw, _ = measure_p2p_bandwidth(src_rank, dst_rank, tensor_size_mb=100)

            if rank == src_rank and bw is not None:
                results[(src_node, dst_node)] = bw
                results[(dst_node, src_node)] = bw  # Assume symmetric

    # Gather results
    all_results = [None] * dist.get_world_size()
    dist.all_gather_object(all_results, results if results else {})

    # Merge
    merged = {}
    for r in all_results:
        if r:
            merged.update(r)

    return merged


def ring_bandwidth_test(tensor_size_mb: int = 100, num_iterations: int = 10) -> float:
    """Simple ring test where each rank sends to next in circle.

    Args:
        tensor_size_mb: Size of tensor to transfer in MB
        num_iterations: Number of iterations

    Returns:
        Average bandwidth across all ranks (GB/s)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    tensor_size = tensor_size_mb * 1024 * 1024 // 4
    tensor = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        send_req = dist.isend(tensor, dst=next_rank)
        recv_req = dist.irecv(tensor, src=prev_rank)
        send_req.wait()
        recv_req.wait()
    end.record()

    end.synchronize()
    elapsed_ms = start.elapsed_time(end)

    bytes_transferred = tensor_size * 4 * num_iterations
    bandwidth_gbps = (bytes_transferred / (elapsed_ms / 1000)) / 1e9

    # Gather all bandwidths
    all_bw = [None] * world_size
    dist.all_gather_object(all_bw, bandwidth_gbps)

    if rank == 0:
        avg_bw = sum(all_bw) / len(all_bw)
        min_bw = min(all_bw)
        max_bw = max(all_bw)
        print(f"Ring test: avg={avg_bw:.2f} GB/s, min={min_bw:.2f}, max={max_bw:.2f}")
        return avg_bw

    return bandwidth_gbps
