# Copyright (c) 2024, Adaptive Pipeline Parallelism Research
# Network topology discovery for multi-node distributed training

import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.distributed as dist


@dataclass
class NodeInfo:
    """Information about a single compute node.

    Attributes:
        node_id: Unique identifier for this node
        hostname: Hostname of the node
        gpu_ranks: List of global ranks of GPUs on this node
        num_gpus: Number of GPUs on this node
    """
    node_id: int
    hostname: str
    gpu_ranks: List[int] = field(default_factory=list)
    num_gpus: int = 0


@dataclass
class NetworkTopology:
    """Complete network topology information.

    This class stores the topology of the distributed system, including
    which GPUs are on which nodes and the measured bandwidth between them.

    Attributes:
        world_size: Total number of GPUs in the system
        num_nodes: Number of compute nodes
        nodes: Dictionary mapping node_id to NodeInfo
        intra_node_bandwidth: Dict mapping node_id to intra-node bandwidth (GB/s)
        inter_node_bandwidth: Dict mapping (src_node, dst_node) to bandwidth (GB/s)
    """
    world_size: int
    num_nodes: int
    nodes: Dict[int, NodeInfo] = field(default_factory=dict)
    intra_node_bandwidth: Dict[int, float] = field(default_factory=dict)
    inter_node_bandwidth: Dict[tuple, float] = field(default_factory=dict)

    def get_node_for_rank(self, rank: int) -> int:
        """Get node ID for a given global rank.

        Args:
            rank: Global rank to query

        Returns:
            Node ID containing this rank

        Raises:
            ValueError: If rank is not found in topology
        """
        for node_id, node_info in self.nodes.items():
            if rank in node_info.gpu_ranks:
                return node_id
        raise ValueError(f"Rank {rank} not found in topology")

    def are_ranks_colocated(self, rank1: int, rank2: int) -> bool:
        """Check if two ranks are on the same node.

        Args:
            rank1: First global rank
            rank2: Second global rank

        Returns:
            True if both ranks are on the same node
        """
        return self.get_node_for_rank(rank1) == self.get_node_for_rank(rank2)

    def get_expected_bandwidth(self, src_rank: int, dst_rank: int) -> Optional[float]:
        """Get expected bandwidth between two ranks.

        Args:
            src_rank: Source rank
            dst_rank: Destination rank

        Returns:
            Expected bandwidth in GB/s or None if not measured
        """
        if self.are_ranks_colocated(src_rank, dst_rank):
            # Intra-node bandwidth
            node_id = self.get_node_for_rank(src_rank)
            return self.intra_node_bandwidth.get(node_id, None)
        else:
            # Inter-node bandwidth
            src_node = self.get_node_for_rank(src_rank)
            dst_node = self.get_node_for_rank(dst_rank)
            return self.inter_node_bandwidth.get((src_node, dst_node), None)

    def print_summary(self, rank: int = 0):
        """Print topology summary (only from specified rank).

        Args:
            rank: Rank that should print (default: 0)
        """
        if dist.get_rank() != rank:
            return

        print("=" * 60)
        print("Network Topology Summary")
        print("=" * 60)
        print(f"World size: {self.world_size} GPUs")
        print(f"Nodes: {self.num_nodes}")
        print()

        for node_id, node_info in sorted(self.nodes.items()):
            print(f"  Node {node_id} ({node_info.hostname}):")
            print(f"    GPUs: {node_info.gpu_ranks}")

            # Print intra-node bandwidth if available
            if node_id in self.intra_node_bandwidth:
                bw = self.intra_node_bandwidth[node_id]
                if bw != float('inf'):
                    print(f"    Intra-node BW: {bw:.2f} GB/s")

        if self.inter_node_bandwidth:
            print()
            print("  Inter-node Bandwidth:")
            for (src, dst), bw in sorted(self.inter_node_bandwidth.items()):
                if src < dst:  # Only print once per pair
                    print(f"    Node {src} <-> Node {dst}: {bw:.2f} GB/s")

        print("=" * 60)


def discover_topology() -> NetworkTopology:
    """Discover network topology from distributed environment.

    This function gathers hostnames from all ranks to determine which
    GPUs are co-located on the same nodes.

    Returns:
        NetworkTopology object with node mapping

    Note:
        Requires torch.distributed to be initialized
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before discovering topology")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Get hostname (unique per node)
    hostname = socket.gethostname()

    # Gather all hostnames to all ranks
    hostname_list = [None] * world_size
    dist.all_gather_object(hostname_list, hostname)

    # Build node mapping
    hostname_to_node_id = {}
    node_id_counter = 0
    nodes = {}

    for r, h in enumerate(hostname_list):
        if h not in hostname_to_node_id:
            # New node discovered
            hostname_to_node_id[h] = node_id_counter
            nodes[node_id_counter] = NodeInfo(
                node_id=node_id_counter,
                hostname=h,
                gpu_ranks=[r],
                num_gpus=1
            )
            node_id_counter += 1
        else:
            # Existing node, add GPU
            node_id = hostname_to_node_id[h]
            nodes[node_id].gpu_ranks.append(r)
            nodes[node_id].num_gpus += 1

    topology = NetworkTopology(
        world_size=world_size,
        num_nodes=len(nodes),
        nodes=nodes,
        intra_node_bandwidth={},
        inter_node_bandwidth={}
    )

    return topology
