# Copyright (c) 2024, Adaptive Pipeline Parallelism Research
# Initial network profiling orchestrator for job startup

import time

import torch.distributed as dist

from .bandwidth_profiler import (
    profile_intra_node_bandwidth,
    profile_inter_node_bandwidth,
    ring_bandwidth_test
)
from .topology import discover_topology, NetworkTopology


def run_initial_profiling(
    enable_profiling: bool = True,
    verbose: bool = True
) -> NetworkTopology:
    """Run initial network profiling at job startup.

    This should be called once at the beginning of training,
    after torch.distributed.init_process_group().

    Args:
        enable_profiling: If False, only discovers topology (no bandwidth tests)
        verbose: Print profiling results

    Returns:
        NetworkTopology object with bandwidth information
    """
    rank = dist.get_rank()
    start_time = time.time()

    if rank == 0 and verbose:
        print("=" * 60)
        print("Starting initial network profiling...")
        print("=" * 60)

    # Phase 1: Topology Discovery
    if rank == 0 and verbose:
        print("\nPhase 1: Discovering topology...")

    topology = discover_topology()

    if rank == 0 and verbose:
        print(f"  World size: {topology.world_size} GPUs")
        print(f"  Nodes: {topology.num_nodes}")
        for node_id, node_info in sorted(topology.nodes.items()):
            print(f"    Node {node_id} ({node_info.hostname}): "
                  f"GPUs {node_info.gpu_ranks}")

    if not enable_profiling:
        if rank == 0 and verbose:
            print("\nProfiling disabled. Returning topology only.")
        return topology

    # Phase 2: Intra-Node Profiling
    if rank == 0 and verbose:
        print("\nPhase 2: Profiling intra-node bandwidth...")

    intra_node_bw = profile_intra_node_bandwidth(topology)
    topology.intra_node_bandwidth = intra_node_bw

    if rank == 0 and verbose:
        for node_id, bw in sorted(intra_node_bw.items()):
            if bw != float('inf'):
                print(f"  Node {node_id}: {bw:.2f} GB/s (NVLink/PCIe)")
            else:
                print(f"  Node {node_id}: Single GPU (no intra-node comm)")

    # Phase 3: Inter-Node Profiling
    if topology.num_nodes > 1:
        if rank == 0 and verbose:
            print("\nPhase 3: Profiling inter-node bandwidth...")

        inter_node_bw = profile_inter_node_bandwidth(topology)
        topology.inter_node_bandwidth = inter_node_bw

        if rank == 0 and verbose:
            for (src, dst), bw in sorted(inter_node_bw.items()):
                if src < dst:  # Only print once per pair
                    print(f"  Node {src} <-> Node {dst}: {bw:.2f} GB/s (IB/Ethernet)")
    else:
        if rank == 0 and verbose:
            print("\nPhase 3: Skipping inter-node profiling (single node)")

    # Phase 4: Ring Test Validation
    if rank == 0 and verbose:
        print("\nPhase 4: Running ring validation test...")

    avg_ring_bw = ring_bandwidth_test()

    elapsed_time = time.time() - start_time

    if rank == 0 and verbose:
        print("\n" + "=" * 60)
        print(f"Initial profiling complete! (elapsed: {elapsed_time:.1f}s)")
        print("=" * 60)

        # Print summary statistics
        if intra_node_bw:
            valid_intra = [bw for bw in intra_node_bw.values() if bw != float('inf')]
            if valid_intra:
                avg_intra = sum(valid_intra) / len(valid_intra)
                print(f"\nBandwidth Summary:")
                print(f"  Intra-node average: {avg_intra:.2f} GB/s")

        if inter_node_bw:
            unique_inter = [bw for (src, dst), bw in inter_node_bw.items() if src < dst]
            if unique_inter:
                avg_inter = sum(unique_inter) / len(unique_inter)
                print(f"  Inter-node average: {avg_inter:.2f} GB/s")
                if valid_intra:
                    ratio = avg_intra / avg_inter
                    print(f"  Intra/Inter ratio: {ratio:.1f}×")

        print("\n" + "=" * 60)

    return topology


def get_p2p_network_stats(p2p_communicator):
    """Get network statistics from P2P communicator if available.

    Args:
        p2p_communicator: P2P communicator instance

    Returns:
        dict: Network statistics or None if monitoring disabled
    """
    if hasattr(p2p_communicator, 'get_network_stats'):
        return p2p_communicator.get_network_stats()
    return None


def print_network_stats(p2p_communicator, rank: int = 0):
    """Print network statistics (useful for debugging).

    Args:
        p2p_communicator: P2P communicator instance
        rank: Rank that should print (default: 0)
    """
    if dist.get_rank() == rank:
        stats = get_p2p_network_stats(p2p_communicator)
        if stats:
            print("=" * 60)
            print("P2P Network Statistics (Runtime Monitoring)")
            print("=" * 60)
            if stats.get('bandwidth_gbps'):
                print(f"Bandwidth: {stats['bandwidth_gbps']:.2f} GB/s")
            if stats.get('latency_ms'):
                print(f"Latency: {stats['latency_ms']:.2f} ms")
            print(f"Total transfers: {stats.get('total_transfers', 0)}")
            print(f"Monitored calls: {stats.get('monitored_calls', 0)} / "
                  f"{stats.get('total_calls', 0)}")
            print(f"Sample rate: {stats.get('sample_rate', 0):.1%}")
            print("=" * 60)
        else:
            print("P2P monitoring is disabled")
