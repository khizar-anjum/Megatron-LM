#!/usr/bin/env python3
"""
Standalone test for network profiling only

This script tests just the initial profiling infrastructure without
requiring a full model setup. Useful for quick validation.

Usage:
    # 2 GPUs on single node:
    torchrun --nproc_per_node=2 test_profiling_only.py

    # 4 GPUs on single node:
    torchrun --nproc_per_node=4 test_profiling_only.py

    # Multi-node with SLURM (8 GPUs across 2 nodes):
    srun -N 2 --ntasks-per-node=4 --gres=gpu:4 python test_profiling_only.py
"""

import os
import sys

import torch
import torch.distributed as dist

# Add Megatron to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from megatron.core.pipeline_parallel.initial_profiling import run_initial_profiling


def main():
    """Main function."""
    # Initialize torch.distributed
    if not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Launched with torchrun
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            # Launched with SLURM
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            local_rank = int(os.environ['SLURM_LOCALID'])
        else:
            print("Error: Cannot determine distributed setup")
            print("Please use torchrun or SLURM to launch this script")
            sys.exit(1)

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 70)
        print(" " * 15 + "Network Profiling Test")
        print("=" * 70)
        print(f"World size: {world_size} GPUs")
        print(f"Backend: {dist.get_backend()}")
        print("=" * 70)

    # Run profiling
    topology = run_initial_profiling(
        enable_profiling=True,
        verbose=(rank == 0)
    )

    # Additional analysis
    if rank == 0:
        print("\n" + "=" * 70)
        print(" " * 20 + "Analysis")
        print("=" * 70)

        # Check for potential issues
        if topology.num_nodes == 1:
            print("✓ Single-node setup detected")
            print("  - All communication will be via NVLink/PCIe")
            print("  - Expected bandwidth: 200-600 GB/s")
        else:
            print(f"✓ Multi-node setup detected ({topology.num_nodes} nodes)")
            print("  - Inter-node communication via InfiniBand/Ethernet")
            print("  - Expected intra-node: 200-600 GB/s")
            print("  - Expected inter-node: 10-25 GB/s")

        # Recommendations
        print("\n" + "-" * 70)
        print("Recommendations for Pipeline Parallelism:")
        print("-" * 70)

        if topology.num_nodes > 1:
            gpus_per_node = topology.nodes[0].num_gpus

            print(f"• Pack {gpus_per_node} pipeline stages per node to minimize")
            print("  inter-node communication")
            print(f"• With {world_size} GPUs, you have {topology.num_nodes} nodes")

            # Calculate good pipeline sizes
            good_sizes = []
            for size in [2, 4, 8, 16, 32]:
                if size <= world_size and world_size % size == 0:
                    stages_per_node = size // topology.num_nodes
                    if stages_per_node <= gpus_per_node:
                        good_sizes.append(size)

            if good_sizes:
                print(f"• Good pipeline parallel sizes: {good_sizes}")
            else:
                print(f"• Consider using {world_size} stages (all GPUs)")

        else:
            print("• Any pipeline parallel size will work well (single node)")
            print(f"• Can use up to {world_size} pipeline stages")

        print("=" * 70)

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print("\n✓ Test completed successfully!")


if __name__ == '__main__':
    main()
